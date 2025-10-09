#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线文档增强（纯本地数据 + 第三方 LLM 推理）：
- 数据：LLukas22/nq-simplified
- 增强：① 文档重写成 atomic facts；② 基于原文合成 QA（答案必须是原文精确子串）
- LLM：方舟/Volc Ark Chat Completions（如 deepseek-v3-1-250821）

输出：
out_dir/
  ├─ aug.jsonl           # 每行: {id, question, context, rewrite_facts: [..], qas: [{q,a}], meta}
  ├─ rewrite/<id>.txt    # 原子事实重写
  ├─ qas/<id>.json       # 通过验真的 QA 列表
  └─ log.txt
"""

import os, re, json, time, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from openai import OpenAI

# ------------------------
# Utils: text normalize & EM/F1
# ------------------------
_ARTICLES = {"a","an","the"}

def white_space_fix(s): return " ".join(s.split())
def remove_articles(s): return " ".join(w for w in s.split() if w not in _ARTICLES)
def remove_punc(s):
    import string
    tbl = str.maketrans("", "", string.punctuation)
    return s.translate(tbl)
def lower(s): return s.lower()

def normalize_answer(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    s = re.sub(r'^[\"\'""`\-–—\(\[]+|[\"\'""`\-–—\)\]]+$', "", s)
    s = s.replace("U.S.", "US").replace("U.K.", "UK")
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = {}
    for t in pred_tokens:
        if t in gold_tokens:
            common[t] = common.get(t, 0) + 1
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def metric_em_f1(pred: str, gold_list: List[str]) -> (float, float):
    if not gold_list: return 0.0, 0.0
    em = max(float(normalize_answer(pred) == normalize_answer(g)) for g in gold_list)
    f1 = max(f1_score(pred, g) for g in gold_list)
    return em, f1

# ------------------------
# Robust JSON extraction
# ------------------------
def extract_json_block(text: str) -> Optional[str]:
    """
    Try to extract a JSON array/object from model output.
    """
    text = text.strip()
    # fenced code block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if m: return m.group(1)
    # first array/object
    m2 = re.search(r"(\[[\s\S]*\])", text)
    if m2: return m2.group(1)
    m3 = re.search(r"(\{[\s\S]*\})", text)
    if m3: return m3.group(1)
    return None

# ------------------------
# LLM Client (Ark / Volcengine)
# ------------------------
class ArkLLM:
    def __init__(self, model_id: str, temperature: float = 0.2):
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise RuntimeError("请先设置环境变量 ARK_API_KEY")
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )
        self.model = model_id
        self.temperature = float(temperature)

    def chat(self, sys: str, user: str, max_tokens: int = 1024) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role":"system","content":sys},
                        {"role":"user","content":user},
                    ],
                    temperature=self.temperature,
                    stream=False,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                # 简单退避重试
                time.sleep(1.5 * (attempt+1))
                if attempt == 2:
                    raise

# ------------------------
# Prompts
# ------------------------
SYS = "You are a careful data annotator. Follow the format strictly."

REWRITE_PROMPT = """You will be given a passage called CONTEXT.
Rewrite it into a concise list of ATOMIC FACTS (bullet list), keeping ALL factual content but removing redundancy.
Rules:
- Each fact ≤ 20 words.
- Preserve original numbers, dates, proper nouns exactly.
- No new facts. No speculation. English only.
- Output JSON array of strings (each item is one fact). Nothing else.

CONTEXT:
{context}
"""

QA_PROMPT = """You will be given a passage called CONTEXT and (optionally) a list of extracted facts.
Generate {k} QA pairs whose answers are the SHORTEST EXACT SUBSTRINGS of CONTEXT.
Rules:
- Answers MUST be exact substrings of CONTEXT.
- Prefer diverse types: entity, number, date, location, short definition.
- If you cannot guarantee an exact substring for a QA, do not include that QA.
- Output ONLY a JSON array; each item: {{"question": "...", "answer": "..."}}. Nothing else.

CONTEXT:
{context}

FACTS (optional, may help you choose better spans):
{facts_json}
"""

# ------------------------
# Core augmentation
# ------------------------
def rewrite_context(llm: ArkLLM, context: str, max_context_chars: int) -> List[str]:
    ctx = context[:max_context_chars]
    out = llm.chat(SYS, REWRITE_PROMPT.format(context=ctx), max_tokens=1024)
    block = extract_json_block(out)
    facts = []
    if block:
        try:
            facts = json.loads(block)
            facts = [s.strip() for s in facts if isinstance(s, str) and s.strip()]
        except Exception:
            facts = []
    # 兜底：若解析失败，退回空（后续可直接用原文）
    return facts

def gen_qas(llm: ArkLLM, context: str, facts: List[str], k: int, max_context_chars: int) -> List[Dict[str,str]]:
    ctx = context[:max_context_chars]
    facts_json = json.dumps(facts[:10], ensure_ascii=False) if facts else "[]"
    out = llm.chat(SYS, QA_PROMPT.format(context=ctx, facts_json=facts_json, k=k), max_tokens=1024)
    block = extract_json_block(out)
    qas = []
    if block:
        try:
            arr = json.loads(block)
            for item in arr:
                q = (item.get("question") or "").strip()
                a = (item.get("answer") or "").strip()
                if q and a:
                    qas.append({"question": q, "answer": a})
        except Exception:
            qas = []
    return qas

def verify_qas(context: str, qas: List[Dict[str,str]], min_f1: float = 0.8) -> List[Dict[str,str]]:
    """
    通过两道闸门：
      1) 答案必须是原文精确子串
      2) 和自身作为金答案的 EM/F1（恒为1/1），这里加入 min_f1 只为防异常空串；同时可拓展多答案时再用
    """
    verified = []
    for qa in qas:
        a = qa["answer"]
        if not a:
            continue
        # must be exact substring of raw context
        if a in context:
            # 轻量过滤：避免过短或过长（比如 1 字或 40+ 字）
            alen = len(a.split())
            if 1 <= alen <= 12:
                # 基本通过；如果还想更严，可以在此加 spanize 再做 EM/F1
                verified.append(qa)
    # 去重（按 question 文本）
    seen = set(); uniq = []
    for qa in verified:
        key = qa["question"].strip().lower()
        if key not in seen:
            seen.add(key); uniq.append(qa)
    return uniq

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="deepseek-v3-1-250821")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--facts_per_doc", type=int, default=6)
    ap.add_argument("--qas_per_doc", type=int, default=6)
    ap.add_argument("--max_context_chars", type=int, default=3800)
    ap.add_argument("--out_dir", type=str, default="aug_nq")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    (out_dir / "rewrite").mkdir(parents=True, exist_ok=True)
    (out_dir / "qas").mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "log.txt", "w", encoding="utf-8")

    def log(s: str):
        print(s)
        log_f.write(s + "\n"); log_f.flush()

    # Load NQ
    split_expr = f"{args.split}[{args.start}:{args.start+args.n}]"
    ds = load_dataset("LLukas22/nq-simplified", split=split_expr)

    llm = ArkLLM(model_id=args.model_id, temperature=args.temperature)

    aug_path = out_dir / "aug.jsonl"
    with open(aug_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(ds):
            q = ex["question"]
            ctx = ex["context"]
            ex_id = args.start + i

            # 1) rewrite to atomic facts
            facts = rewrite_context(llm, ctx, args.max_context_chars)
            # fallback: 若无 facts，退回空列表（后续训练可直接选择用原文重建）
            if args.verbose:
                log(f"[{ex_id}] facts={len(facts)}")

            # 2) synthesize QA pairs (答案为原文子串)
            raw_qas = gen_qas(llm, ctx, facts, k=args.qas_per_doc, max_context_chars=args.max_context_chars)
            qas = verify_qas(ctx, raw_qas)
            if args.verbose:
                log(f"[{ex_id}] qas_raw={len(raw_qas)} qas_verified={len(qas)}")

            # save side files
            with open(out_dir / "rewrite" / f"{ex_id}.txt", "w", encoding="utf-8") as fr:
                fr.write("\n".join(facts) if facts else "")
            with open(out_dir / "qas" / f"{ex_id}.json", "w", encoding="utf-8") as fq:
                json.dump(qas, fq, ensure_ascii=False, indent=2)

            # main jsonl line
            rec = {
                "id": int(ex_id),
                "question": q,
                "context": ctx,
                "rewrite_facts": facts,
                "qas": qas,
                "meta": {
                    "facts_per_doc": args.facts_per_doc,
                    "qas_per_doc": args.qas_per_doc,
                    "max_context_chars": args.max_context_chars,
                    "model_id": args.model_id,
                    "temperature": args.temperature,
                }
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log("Done.")
    log_f.close()

if __name__ == "__main__":
    main()
