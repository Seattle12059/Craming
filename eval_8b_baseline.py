#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, re, string, random
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# SQuAD-style normalization & metrics
# =========================
_ARTICLES = {"a", "an", "the"}

def white_space_fix(s): return " ".join(s.split())
def remove_articles(s): return " ".join([w for w in s.split() if w not in _ARTICLES])
def remove_punc(s):
    tbl = str.maketrans("", "", string.punctuation)
    return s.translate(tbl)
def lower(s): return s.lower()

def normalize_answer(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    # 去掉引号/括号等常见包裹符
    s = re.sub(r'^[\"\'`\-–—\(\[]+|[\"\'`\-–—\)\]]+$', "", s)
    # 常见缩写统一（可按需扩展）
    s = s.replace("U.S.", "US").replace("U.K.", "UK")
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    # token 交集计数
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

def metric_em_f1(pred: str, gold_list: List[str]) -> Tuple[float, float]:
    if not gold_list: return 0.0, 0.0
    em = max(float(normalize_answer(pred) == normalize_answer(g)) for g in gold_list)
    f1 = max(f1_score(pred, g) for g in gold_list)
    return em, f1

# =========================
# Prompt helpers
# =========================
SYS_PROMPT = "You are a helpful, concise QA assistant."

def build_prompt(tokenizer, question: str, context: str = None) -> str:
    if context:
        user = (
            "From the given context, COPY the shortest exact answer span.\n"
            "If the answer is not in the context, output exactly: I don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer (span only):"
        )
    else:
        user = (
            "Answer the question with a short fact (≤3 words). No punctuation.\n"
            f"Question: {question}\nAnswer:"
        )
    # 如果是指令/对话模型，优先走 chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # fallback: 简单 instruction 格式
    return f"### System:\n{SYS_PROMPT}\n\n### User:\n{user}\n\n### Assistant:"

def postprocess_model_answer(text: str) -> str:
    ans = text.strip().splitlines()[0].strip()
    ans = re.sub(r"^(answer\s*:)\s*", "", ans, flags=re.I)
    ans = re.sub(r"[\"“”‘’]+$", "", ans).strip()
    return ans

# =========================
# Generation
# =========================


@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens=16, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature <= 0 else True,
        temperature=temperature if temperature > 0 else None,
        top_p=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return postprocess_model_answer(text)

# =========================
# NON-ORACLE fallback: 只根据“模型预测”在 context 里对齐一个最像的短片段（不看 gold）
# =========================
def _f1_tokens(pred_tokens, span_tokens):
    # 与 f1_score 相同思路，但对象是两个 token 列表
    common = {}
    for t in span_tokens:
        if t in pred_tokens:
            common[t] = common.get(t, 0) + 1
    num_same = sum(min(span_tokens.count(t), pred_tokens.count(t)) for t in common)
    if len(span_tokens) == 0 or len(pred_tokens) == 0:
        return float(span_tokens == pred_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(span_tokens)
    recall = num_same / len(pred_tokens)
    return 2 * precision * recall / (precision + recall)

def spanize_to_context(pred: str, context: str, max_len: int = 8) -> str:
    # 如果预测文本就是上下文的一段，直接用它
    if pred and pred in context:
        return pred
    pred_tokens = normalize_answer(pred).split()
    if not pred_tokens:
        return pred
    tokens = context.split()
    best = pred
    best_score = -1.0
    for L in range(1, min(max_len, len(tokens)) + 1):
        for i in range(0, len(tokens) - L + 1):
            span = " ".join(tokens[i:i+L])
            score = _f1_tokens(pred_tokens, normalize_answer(span).split())
            if score > best_score:
                best_score, best = score, span
    return best

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model path or local dir (e.g., Llama 8B Instruct)")
    ap.add_argument("--n", type=int, default=100, help="number of examples")
    ap.add_argument("--split", type=str, default="test", help="dataset split: train/test")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)   # 抽取式建议更短
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv_out", type=str, default="preds_nq_llama8b.csv")
    ap.add_argument("--save_preds", type=str, default=None, help="optional JSONL path")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","auto"], help="cuda for single-GPU, auto for HF device_map")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset（SQuAD-like）：LLukas22/nq-simplified
    split_expr = f"{args.split}[:{args.n}]"
    ds = load_dataset("LLukas22/nq-simplified", split=split_expr)

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda" if args.device == "cuda" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # CSV 列：第一列 = context_len_tokens
    header = [
        "context_len_tokens","id","question","context","gold_answers",
        "pred_no_ctx","pred_with_ctx","em_no_ctx","f1_no_ctx","em_with_ctx","f1_with_ctx",
    ]

    em_1 = f1_1 = em_2 = f1_2 = 0.0
    rows = []
    jsonl = [] if args.save_preds else None

    for i, ex in enumerate(ds):
        q = ex["question"]
        ctx = ex["context"]
        golds = ex.get("answers", {}).get("text", [])
        gold_str = " || ".join(golds)

        # context token length（不含特殊符号）
        ctx_len_tokens = len(tokenizer.encode(ctx, add_special_tokens=False))

        # (1) no context
        prompt1 = build_prompt(tokenizer, q, context=None)
        pred1 = generate_answer(model, tokenizer, prompt1, args.max_new_tokens, args.temperature)
        em1, f11 = metric_em_f1(pred1, golds)

        # (2) with context（非 oracle 对齐）
        prompt2 = build_prompt(tokenizer, q, context=ctx)
        pred2 = generate_answer(model, tokenizer, prompt2, args.max_new_tokens, args.temperature)
        pred2_span = spanize_to_context(pred2, ctx)  # 不看 gold，只把预测对齐为上下文中的短片段
        em2, f12 = metric_em_f1(pred2_span, golds)

        if args.verbose and (i+1) % 10 == 0:
            print(f"[{i+1}/{args.n}] EM/F1(no-ctx)={em1:.0f}/{f11:.1f}  EM/F1(+ctx)={em2:.0f}/{f12:.1f}")

        em_1 += em1; f1_1 += f11; em_2 += em2; f1_2 += f12

        rows.append([
            ctx_len_tokens, i, q, ctx, gold_str,
            pred1, pred2_span,
            f"{em1:.3f}", f"{f11:.3f}", f"{em2:.3f}", f"{f12:.3f}",
        ])

        if jsonl is not None:
            jsonl.append({
                "id": i,
                "question": q,
                "context": ctx,
                "gold_answers": golds,
                "pred_no_ctx": pred1,
                "pred_with_ctx": pred2_span,
                "em_no_ctx": em1, "f1_no_ctx": f11,
                "em_with_ctx": em2, "f1_with_ctx": f12,
                "context_len_tokens": ctx_len_tokens,
            })

    N = max(1, len(rows))
    print("\n=== Final ===")
    print(f"(1) No context     : EM={em_1/N*100:.1f}  F1={f1_1/N*100:.1f}")
    print(f"(2) With gold ctx  : EM={em_2/N*100:.1f}  F1={f1_2/N*100:.1f}")
    print(f"Saving CSV -> {args.csv_out}")

    # Save CSV
    with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Optional JSONL
    if jsonl is not None:
        with open(args.save_preds, "w", encoding="utf-8") as f:
            for r in jsonl:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL -> {args.save_preds}")

if __name__ == "__main__":
    main()
