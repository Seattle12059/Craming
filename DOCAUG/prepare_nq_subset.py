#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, random, re
from typing import List, Tuple, Dict
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def coarse_answer_type(ans: str) -> str:
    a = ans.strip()
    if re.fullmatch(r"\d{1,4}(-\d{1,2}(-\d{1,2})?)?", a): return "date"
    if re.fullmatch(r"[0-9,\.]+", a): return "number"
    if re.search(r"\b(USA|UK|China|France|New York|London|Paris|Tokyo|India)\b", a, re.I): return "loc"
    if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", a): return "person"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train","test"])
    ap.add_argument("--out_dir", default="data/nq_subset")
    ap.add_argument("--subset_size", type=int, default=100)
    ap.add_argument("--min_ctx_toks", type=int, default=500)
    ap.add_argument("--max_ctx_toks", type=int, default=1200)
    ap.add_argument("--tokenizer_name", default="gpt2", help="仅用于长度统计，不影响后续模型")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)

    print(f"Loading dataset LLukas22/nq-simplified split={args.split} ...")
    ds = load_dataset("LLukas22/nq-simplified", split=args.split)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    def ok_example(ex):
        golds = ex.get("answers", {}).get("text", [])
        if not golds: return False
        ctx = ex["context"]
        # 答案必须出现于上下文（保证可抽取）
        if not any(g and g in ctx for g in golds):
            return False
        # 统计上下文 token 长度
        n_tok = len(tok(ctx, add_special_tokens=False)["input_ids"])
        return (args.min_ctx_toks <= n_tok <= args.max_ctx_toks)

    print("Filtering by constraints (answer in context, 500<=ctx_tokens<=1200) ...")
    ds_f = ds.filter(ok_example)

    print(f"Filtered size: {len(ds_f)}")

    # 增加元数据（答案类型、长度分桶）用于分层采样
    def add_meta(ex):
        golds = ex["answers"]["text"]
        # 取最长答案作为代表（更稳定）
        ans = max(golds, key=len)
        ex["answer_repr"] = ans
        ex["ans_type"] = coarse_answer_type(ans)
        ex["ctx_len_chars"] = len(ex["context"])
        return ex

    ds_f = ds_f.map(add_meta)

    def bucket_len(L):
        if L < 800: return "short"
        if L < 2000: return "mid"
        return "long"

    # 分层采样：按“答案类型 × 长度桶”均衡取样
    buckets: Dict[Tuple[str,str], List[dict]] = {}
    for ex in ds_f:
        key = (ex["ans_type"], bucket_len(ex["ctx_len_chars"]))
        buckets.setdefault(key, []).append(ex)

    target_total = args.subset_size
    keys = list(buckets.keys())
    random.shuffle(keys)
    per_bucket = max(1, target_total // max(1, len(keys)))

    selected: List[dict] = []
    for k in keys:
        items = buckets[k]
        random.shuffle(items)
        selected.extend(items[:min(per_bucket, len(items))])

    # 如果不足 100，再从剩余里补齐
    if len(selected) < target_total:
        pool = [ex for k in keys for ex in buckets[k] if ex not in selected]
        random.shuffle(pool)
        selected.extend(pool[:target_total - len(selected)])

    # 去重（避免同question/context重复过近）
    seen = set()
    final = []
    for ex in selected:
        sig = (ex["question"].strip(), ex["context"][:500])
        if sig in seen: continue
        seen.add(sig); final.append(ex)
        if len(final) >= target_total: break

    out_file = os.path.join(args.out_dir, "raw", "nq_subset_100_dif.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in final:
            rec = {
                "id": ex.get("id"),
                "question": ex["question"],
                "context": ex["context"],
                "answers": ex["answers"]["text"],
                "answer_repr": ex["answer_repr"],
                "ans_type": ex["ans_type"]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(final)} -> {out_file}")

if __name__ == "__main__":
    main()
