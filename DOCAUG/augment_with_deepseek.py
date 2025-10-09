#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 DeepSeek (Ark) 的 NQ 文档增强脚本
- 当前默认：仅基于原始 context 生成多个新 QA（不重写、不释义、不造不可回答）。
- 未来：支持上下文重写、问题释义、不可回答问题；可通过参数开启。

输入（每行一个样本，来自 prepare_nq_subset.py 的输出）：
{
  "id": ...,
  "question": "...",
  "context": "...",
  "answers": ["...", "..."],
  "answer_repr": "...",
  "ans_type": "..."
}

输出（每行一个样本）：
{
  "id": ...,
  "context_orig": "...",           # 原始上下文
  "context_aug": [...],            # 可选的重写版本（默认空）
  "qas": [                         # 新生成/保留的问答
    {"q":"...","a":"...","a_span":true/false}
  ],
  "stats": {                       # 简单统计，便于核对
    "ctx_in": 1,
    "ctx_out": 1 + rewrites_per_context,
    "qa_in": 0 或 1（取决于是否 drop 原问题）,
    "qa_out": 最终 QA 数
  }
}

用法（满足你当前的“只生成多个问题”的需求）：
------------------------------------------------
python -u augment_with_deepseek.py \
  --in_file data/nq_subset/raw/nq_subset_100.jsonl \
  --out_file data/nq_subset/augmented/nq_multiqa_only.jsonl \
  --new_qas_per_context 5 \
  --drop_original_q \
  --rewrites_per_context 0 \
  --paraphrases_per_q 0 \
  --unans_per_context 0
"""

import argparse
import json
import os
import random
from typing import List, Dict, Tuple

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError


# -----------------------
# DeepSeek / Ark client
# -----------------------
MODEL_NAME = "deepseek-v3-1-terminus"

def build_client():
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    if not api_key:
        raise RuntimeError("缺少环境变量 ARK_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url)


@retry(
    retry=retry_if_exception_type((APIConnectionError, APIStatusError, RateLimitError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5),
    reraise=True,
)
def call_chat(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        stream=False,
    )
    return resp.choices[0].message.content.strip()


# -----------------------
# Prompts
# -----------------------
SYS_GEN_QA = (
    "You are an information extraction assistant. "
    "Generate STRICT JSONL (one JSON per line). "
    "Answers MUST be the SHORTEST exact spans from the text. "
    "Do NOT output any commentary."
)
TPL_GEN_QA = (
    "Text:\n{ctx}\n\n"
    "Generate {K} extractive QA pairs for the text above.\n"
    "Each line MUST be a single JSON object with fields exactly as follows:\n"
    '{{"q":"<question>","a":"<shortest exact span answer from text>","a_is_exact_span":true}}\n'
    "Constraints:\n"
    "- Questions must be answerable only using the text.\n"
    "- Answers must be the SHORTEST exact spans copied verbatim from the text.\n"
    "- Avoid yes/no questions and avoid duplicating the original question: {orig_q}\n"
    "- Provide {K} lines exactly. No extra text."
)

# 备用：上下文重写 / 问题释义 / 不可回答（预留未来用）
SYS_PARAPHRASE_CTX = (
    "You are a precise data augmenter. Rewrite the given English passage, "
    "but you MUST keep the exact answer string unchanged. Do not add new facts."
)
TPL_PARAPHRASE_CTX = (
    "Answer string to PRESERVE exactly:\n{ans}\n\n"
    "Rewrite the following passage in English. Output ONLY the rewritten passage:\n\n{ctx}"
)

SYS_PARAPHRASE_Q = "You paraphrase questions in English while keeping the answer unchanged."
TPL_PARAPHRASE_Q = (
    "Given the original question and the answer string (must remain consistent), "
    "produce {N} paraphrases in English. Output ONE question per line. No numbering.\n\n"
    "Answer:\n{ans}\n\nQuestion:\n{q}"
)

SYS_UNANS = "You create unanswerable-but-related questions in English. The answer must NOT be in the text."
TPL_UNANS = (
    "Given the text below, create {N} unanswerable questions that are related to its topic but cannot be answered using the text. "
    "Output ONE question per line. No numbering.\n\n{ctx}"
)


# -----------------------
# Helpers
# -----------------------
def parse_jsonl_lines(s: str) -> List[Dict]:
    """Parse multiple lines of JSON; ignore lines that cannot be parsed."""
    out = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out.append(obj)
        except Exception:
            continue
    return out


def gen_qa_from_context(client: OpenAI, context: str, k: int, orig_q: str = "", max_answer_len: int = 120) -> List[Dict]:
    """Generate k extractive QA pairs from context. Ensure answers are exact spans and not too long."""
    raw = call_chat(client, SYS_GEN_QA, TPL_GEN_QA.format(ctx=context, K=k, orig_q=(orig_q or "<none>")))
    items = parse_jsonl_lines(raw)

    qas: List[Dict] = []
    for it in items:
        q = (it.get("q") or "").strip()
        a = (it.get("a") or "").strip()
        span = bool(it.get("a_is_exact_span", True))
        if not q or not a or not span:
            continue
        if a not in context:
            continue
        if len(a) > max_answer_len:
            continue
        qas.append({"q": q, "a": a, "a_span": True})
    return qas


def paraphrase_context_keep_answer(client: OpenAI, context: str, answer: str, n: int) -> List[str]:
    outs = []
    for _ in range(n):
        text = call_chat(client, SYS_PARAPHRASE_CTX, TPL_PARAPHRASE_CTX.format(ctx=context, ans=answer))
        if answer in text:
            outs.append(text)
    return outs


def paraphrase_question_keep_answer(client: OpenAI, q: str, ans: str, n: int) -> List[str]:
    raw = call_chat(client, SYS_PARAPHRASE_Q, TPL_PARAPHRASE_Q.format(N=n, ans=ans, q=q))
    outs = [line.strip() for line in raw.splitlines() if line.strip()]
    return outs


def build_unanswerable(client: OpenAI, context: str, n: int) -> List[Dict]:
    raw = call_chat(client, SYS_UNANS, TPL_UNANS.format(N=n, ctx=context))
    outs = [{"q": line.strip(), "a": "I don't know.", "a_span": False}
            for line in raw.splitlines() if line.strip()]
    return outs


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="data/nq_subset/raw/nq_subset_100.jsonl")
    ap.add_argument("--out_file", default="data/nq_subset/augmented/nq_multiqa_only.jsonl")

    # 你当前只需要 new_qas_per_context；其余开关默认 0 即可
    ap.add_argument("--new_qas_per_context", type=int, default=5, help="每条样本基于原始 context 生成的新 QA 数量")
    ap.add_argument("--drop_original_q", action="store_true", help="不把原问题写入输出（保留给评测）")

    # 下面三个是为以后多样增强预留；当前保持 0
    ap.add_argument("--rewrites_per_context", type=int, default=0, help="上下文重写次数（每次保留答案原文）")
    ap.add_argument("--paraphrases_per_q", type=int, default=0, help="原问题释义数量")
    ap.add_argument("--unans_per_context", type=int, default=0, help="不可回答问题个数")

    ap.add_argument("--max_answer_len", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    in_path = os.path.abspath(os.path.expanduser(args.in_file))
    out_path = os.path.abspath(os.path.expanduser(args.out_file))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 预检
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"[precheck] Input file not found: {in_path}")
    if os.path.getsize(in_path) == 0:
        raise RuntimeError(f"[precheck] Input file is empty: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"[precheck] in_file={in_path}")
    print(f"[precheck] out_file={out_path}")
    print(f"[precheck] lines={total_lines}")

    # 是否需要 LLM（当前你的配置只依赖 new_qas_per_context）
    need_llm = any([
        args.new_qas_per_context > 0,
        args.rewrites_per_context > 0,
        args.paraphrases_per_q > 0,
        args.unans_per_context > 0
    ])
    client = build_client() if need_llm else None
    if not need_llm:
        print("[run] No augmentation requested; nothing to do.")
        return

    n_in = n_out = 0
    total_qas = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=total_lines, desc="augment")
        for line in fin:
            pbar.update(1)
            n_in += 1
            ex = json.loads(line)

            ctx = ex.get("context") or ex.get("context_orig")
            if not ctx:
                continue

            # 取原答案的代表；用于可选的重写/释义；对“只生成新问”不是硬依赖
            golds = ex.get("answers") or []
            ans_repr = ex.get("answer_repr") or (golds[0] if golds else "")

            # 原问题（可选地丢弃）
            orig_q = (ex.get("question") or "").strip()

            # 1) 上下文重写（当前默认 0）
            ctx_augs: List[str] = []
            if args.rewrites_per_context > 0 and ans_repr and ans_repr in ctx:
                ctx_augs = paraphrase_context_keep_answer(client, ctx, ans_repr, args.rewrites_per_context)

            # 2) 生成新问答（只针对原始 context；如果你以后想包含重写文本，改成 sources=[ctx]+ctx_augs）
            qas: List[Dict] = []
            if args.new_qas_per_context > 0:
                qas.extend(gen_qa_from_context(client, ctx, args.new_qas_per_context, orig_q=orig_q, max_answer_len=args.max_answer_len))

            # 3) 原问题（可选保留/丢弃 —— 你当前需求是 drop）
            qa_in_count = 0
            if orig_q and (not args.drop_original_q):
                # 若保留原问题，需要一个对应答案；这里用代表答案（不强制 exact span）
                # 如需严格 exact span，可在此做 span 对齐
                if ans_repr:
                    qas.append({"q": orig_q, "a": ans_repr, "a_span": (ans_repr in ctx)})
                    qa_in_count = 1

            # 4) 原问题释义（默认 0）
            if args.paraphrases_per_q > 0 and orig_q and ans_repr:
                for pq in paraphrase_question_keep_answer(client, orig_q, ans_repr, args.paraphrases_per_q):
                    qas.append({"q": pq, "a": ans_repr, "a_span": (ans_repr in ctx)})

            # 5) 不可回答（默认 0）
            if args.unans_per_context > 0:
                qas.extend(build_unanswerable(client, ctx, args.unans_per_context))

            # 去重（按 问题文本 + 答案 + a_span）
            seen = set()
            qas_dedup = []
            for qa in qas:
                sig = (qa["q"].strip(), qa["a"].strip(), bool(qa["a_span"]))
                if sig in seen:
                    continue
                seen.add(sig)
                qas_dedup.append(qa)

            rec = {
                "id": ex.get("id"),
                "context_orig": ctx,
                "context_aug": ctx_augs,  # 现在通常为空列表
                "qas": qas_dedup,
                "stats": {
                    "ctx_in": 1,
                    "ctx_out": 1 + len(ctx_augs),
                    "qa_in": qa_in_count,
                    "qa_out": len(qas_dedup)
                }
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
            total_qas += len(qas_dedup)
        pbar.close()

    print(f"[done] read={n_in} wrote={n_out}  avg_qas_per_doc={total_qas / max(1,n_out):.2f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

"""
export ARK_API_KEY="b44e3b9c-7463-4284-826a-efc75f4aeef3"
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"

python -u augment_with_deepseek.py \
  --in_file data/nq_subset/raw/nq_subset_100.jsonl \
  --out_file data/nq_subset/augmented/nq_multiqa_only_no_unans_first100.jsonl \
  --new_qas_per_context 10 \
  --drop_original_q \
  --rewrites_per_context 0 \
  --paraphrases_per_q 0 \
  --unans_per_context 0


"""