#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from typing import List
from datasets import load_dataset

def coarse_answer_type(ans: str) -> str:
    a = (ans or "").strip()
    if not a: return "other"
    if re.fullmatch(r"\d{1,4}(-\d{1,2}(-\d{1,2})?)?", a): return "date"
    if re.fullmatch(r"[0-9,\.]+", a): return "number"
    if re.search(r"\b(USA|UK|China|France|New York|London|Paris|Tokyo|India)\b", a, re.I): return "loc"
    if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", a): return "person"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/nq_subset")
    ap.add_argument("--subset_name", default="nq_subset_100.jsonl",
                    help="输出文件名，默认与之前保持一致")
    ap.add_argument("--hf_subset", default="test[:100]",
                    help="HF 切片表达式，默认 test[:100]")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)
    out_file = os.path.join(args.out_dir, "raw", args.subset_name)

    print(f"Loading dataset LLukas22/nq-simplified split={args.hf_subset} ...")
    ds = load_dataset("LLukas22/nq-simplified", split=args.hf_subset)

    n = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in ds:
            question = ex.get("question", "")
            context  = ex.get("context", "")
            # 数据集里 answers 形如 {"text": [...]}，兼容兜底
            answers_list: List[str] = []
            ans_field = ex.get("answers")
            if isinstance(ans_field, dict):
                answers_list = ans_field.get("text", []) or []
            elif isinstance(ans_field, list):
                answers_list = ans_field
            answer_repr = max(answers_list, key=len) if answers_list else ""
            ans_type = coarse_answer_type(answer_repr)

            rec = {
                "id": ex.get("id"),
                "question": question,
                "context": context,
                "answers": answers_list,
                "answer_repr": answer_repr,
                "ans_type": ans_type
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Saved {n} -> {out_file}")

if __name__ == "__main__":
    main()
