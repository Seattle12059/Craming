#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, string, sys, random, math
from itertools import islice
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Utils: normalization & metrics (SQuAD-style)
# ----------------------------
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
    # 常见引导词/收尾噪声
    s = re.sub(r"^[\"'`\-–—\(\[]+|[\"'`\-–—\)\]]+$", "", s)
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

def metric_em_f1(pred: str, gold_list: List[str]) -> Tuple[float, float]:
    if not gold_list: return 0.0, 0.0
    em = max(float(normalize_answer(pred) == normalize_answer(g)) for g in gold_list)
    f1 = max(f1_score(pred, g) for g in gold_list)
    return em, f1

# ----------------------------
# Prompt helpers
# ----------------------------
SYS_PROMPT = "You are a helpful, concise QA assistant. Answer with a short fact if possible."

def build_prompt(tokenizer, question: str, context: str = None) -> str:
    if context:
        user = (
            "Use the given context to answer the question with a short fact. "
            "If the answer is not in the context, say \"I don't know\".\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    else:
        user = f"Answer the question with a short fact.\nQuestion: {question}\nAnswer:"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # fallback: simple instruction format
    return f"### System:\n{SYS_PROMPT}\n\n### User:\n{user}\n\n### Assistant:"

def postprocess_model_answer(text: str) -> str:
    # 取第一行，去掉多余符号/前缀
    ans = text.strip().splitlines()[0].strip()
    ans = re.sub(r"^(answer\s*:)\s*", "", ans, flags=re.I)
    ans = re.sub(r"[\"“”‘’]+$", "", ans).strip()
    return ans

# ----------------------------
# Inference
# ----------------------------
@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens=64, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return postprocess_model_answer(text)

# ----------------------------
# Main eval
# ----------------------------
def run_eval(args):
    # set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dataset: LLukas22/nq-simplified  (question/context/answers)
    # 选 n 条：默认直接子切片；如网络不佳，可用 streaming=True + islice 采样
    split_expr = f"train[:{args.n}]"
    ds = load_dataset("LLukas22/nq-simplified", split=split_expr)

    # 模型
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,  # 新版参数名；旧版用 torch_dtype=torch.float16
        device_map="cuda",  # 强制放到 GPU
        low_cpu_mem_usage=True
    )
    results = []
    em_1 = f1_1 = em_2 = f12 = 0.0

    for i, ex in enumerate(ds):
        q = ex["question"]
        ctx = ex["context"] if args.use_context else None
        golds = ex.get("answers", {}).get("text", [])
        # run no-context
        prompt1 = build_prompt(tokenizer, q, context=None)
        pred1 = generate_answer(model, tokenizer, prompt1, args.max_new_tokens, args.temperature)
        em1, f11 = metric_em_f1(pred1, golds)

        # run with context
        prompt2 = build_prompt(tokenizer, q, context=ex["context"])
        pred2 = generate_answer(model, tokenizer, prompt2, args.max_new_tokens, args.temperature)
        em2, f12_i = metric_em_f1(pred2, golds)

        results.append({
            "id": i,
            "question": q,
            "context": ex["context"],
            "gold_answers": golds,
            "pred_no_ctx": pred1,
            "pred_with_ctx": pred2,
            "em_no_ctx": em1, "f1_no_ctx": f11,
            "em_with_ctx": em2, "f1_with_ctx": f12_i,
        })

        em_1 += em1; f1_1 += f11; em_2 += em2; f12 += f12_i

        if args.verbose and (i+1) % 10 == 0:
            print(f"[{i+1}/{args.n}] EM/F1(no-ctx)={em1:.0f}/{f11:.1f}  EM/F1(+ctx)={em2:.0f}/{f12_i:.1f}")

    N = len(results) or 1
    avg_no_ctx = {"EM": em_1 / N, "F1": f1_1 / N}
    avg_with_ctx = {"EM": em_2 / N, "F1": f12 / N}

    print("\n=== Final ===")
    print(f"(1) No context     : EM={avg_no_ctx['EM']*100:.1f}  F1={avg_no_ctx['F1']*100:.1f}")
    print(f"(2) With gold ctx  : EM={avg_with_ctx['EM']*100:.1f}  F1={avg_with_ctx['F1']*100:.1f}")

    if args.save_preds:
        with open(args.save_preds, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved predictions to: {args.save_preds}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="HF model path, e.g. meta-llama/Llama-3.1-8B-Instruct or a local dir")
    ap.add_argument("--n", type=int, default=100, help="number of samples")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_preds", type=str, default="preds_nq.jsonl")
    ap.add_argument("--verbose", action="store_true")
    # for completeness, a switch that turns off context run (not used by default)
    ap.add_argument("--use_context", action="store_true",
                    help="if set, ALSO run context setting (the script always compares ① vs ②)")
    args = ap.parse_args()

    # 我们总是同时评 ①与②，这个 flag 只是为了在 build_prompt 里留钩子
    args.use_context = True
    run_eval(args)

if __name__ == "__main__":
    main()
