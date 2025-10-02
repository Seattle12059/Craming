#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate 1-token memory compression on 10 samples using the author's code.

- Uses the author's MemoryCell (model.py) and training routine (train.run_single_experiment)
- Base model: local LLaMA 3 8B at /home/syt/prjct_folder/llama3_1_8b  (override with --model_path)
- Dataset: author's PG-19 valid chunks CSV (default: ./data/pg19_valid_1k_chunks.csv)
- For each of 10 samples:
    1) Train a single learnable memory token (K=1) to reconstruct a 512-token suffix.
    2) Greedy-generate from [mem] + <BOS> only (no text prompt) to produce "predicate text".
    3) Compute the token-level match ratio vs the original 512-token target.
- Output: CSV with columns [origin_text, predicate_text, match_ratio] (plus metadata).

Usage example:
    python mem_eval_llama3_8b_len512_k1.py \
        --model_path /home/syt/prjct_folder/llama3_1_8b \
        --texts_path ./data/pg19_valid_1k_chunks.csv \
        --num_samples 10 \
        --output_csv ./runs/mem_eval_llama3_8b_len512_k1.csv \
        --dtype bfloat16 \
        --iterations 5000 \
        --patience 2000 \
        --use_flash_attention_2
"""
import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Import author's modules
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import MemoryCell
from train import run_single_experiment  # we'll reuse the author's training loop
from nltk import sent_tokenize

# ---- Helpers ----

def ensure_nltk_punkt():
    """Ensure NLTK punkt is available for sent_tokenize (author's code relies on it)."""
    try:
        _ = sent_tokenize("test.")
    except LookupError:
        import nltk
        nltk.download("punkt")
        _ = sent_tokenize("test.")

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def tokenize_suffix_half(text: str):
    """Split text by sentences and take the *second half* (author's training uses suffix only)."""
    sents = sent_tokenize(text)
    suffix_text = " ".join(sents[len(sents)//2:])
    return suffix_text

def get_memory_dim_from_model(model) -> int:
    return getattr(model.config, "word_embed_proj_dim", getattr(model.config, "hidden_size"))

def build_generation_prompt(tokenizer):
    """
    Return a minimal, content-free BOS-only input_ids so generation is driven by [mem] alone.
    """
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        # Fallback: many LLaMA-style tokenizers use 1 as BOS
        bos_id = 1
    input_ids = torch.tensor([[bos_id]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

def ids_to_text(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def compute_match_ratio(target_ids, gen_ids):
    """
    Token-level accuracy over the target length (gen shorter part counts as mismatches).
    """
    T = len(target_ids)
    G = len(gen_ids)
    matches = 0
    for i in range(T):
        if i < G and gen_ids[i] == target_ids[i]:
            matches += 1
    return matches / max(1, T)

# ---- Main evaluation ----

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate 1-token memory compression on 10 samples")
    ap.add_argument("--model_path", type=str, default="/home/syt/prjct_folder/llama3_1_8b",
                    help="Path to the local HF model directory for LLaMA 3 8B")
    ap.add_argument("--texts_path", type=str, default="./data/pg19_valid_1k_chunks.csv",
                    help="Path to the author's PG-19 chunks CSV file")
    ap.add_argument("--output_csv", type=str, default="./runs/mem_eval_llama3_8b_len512_k1.csv",
                    help="Where to save the evaluation CSV")
    ap.add_argument("--num_samples", type=int, default=10, help="How many rows from the dataset to evaluate")
    ap.add_argument("--iterations", type=int, default=5000, help="Max training steps per sample")
    ap.add_argument("--patience", type=int, default=2000, help="Early stopping patience (no acc improvement)")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float32","float16","bfloat16"],
                    help="Computation dtype for AMP")
    ap.add_argument("--use_flash_attention_2", action="store_true", help="Enable FlashAttention-2 if available")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_nltk_punkt()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    dtype_obj = getattr(torch, args.dtype)

    # Load tokenizer once (author loads per-call inside run_single_experiment too, but that's fine)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Many LLaMA tokenizers don't define pad; set a benign pad id if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Read dataset
    df = pd.read_csv(args.texts_path, index_col=0)
    texts = df["text"].tolist()[:args.num_samples]

    # Consistent hyperparams with the request
    MODEL_NAME = args.model_path
    MAX_LEN = 512
    N_MEM = 1
    NUM_ITERS = args.iterations
    PATIENCE = args.patience
    LR = 1e-2
    BETA1 = 0.9
    BETA2 = 0.9
    WD = 0.01

    rows = []
    pbar = tqdm(range(len(texts)), desc="Evaluating samples", leave=False)

    for idx, full_text in enumerate(texts):
        pbar.update(1)

        # Build the sample's suffix (author's convention)
        suffix_text = tokenize_suffix_half(full_text)

        # Run the author's training loop ONCE to learn best memory for this sample
        result = run_single_experiment(
            N_mem_tokens=N_MEM,
            text_sample=full_text,              # author function will rebuild suffix internally
            max_length=MAX_LEN,
            num_iterations=NUM_ITERS,
            sample_idx=idx,
            run_idx=0,
            model_name=MODEL_NAME,
            dtype=dtype_obj,
            use_flash_attention_2=args.use_flash_attention_2,
            device=device,
            tokenizer=tokenizer,
            lr=LR, beta_1=BETA1, beta_2=BETA2,
            weight_decay=WD,
            early_stopping_patience=PATIENCE,
            shuffled=False
        )

        # Recover the best memory parameters
        best_mem_np = result["best_memory_params"]
        if best_mem_np is None:
            # If no improvement was recorded, fall back to the current memory (rare)
            best_mem_np = np.zeros((N_MEM, get_memory_dim_from_model(
                AutoModelForCausalLM.from_pretrained(MODEL_NAME).config
            )), dtype=np.float32)

        # Rebuild tokenizerized target (WITHOUT special tokens) to define "origin text"
        target_enc = tokenizer(
            result["suffix_text"], max_length=MAX_LEN, truncation=True,
            return_tensors="pt", add_special_tokens=False
        )
        target_ids = target_enc["input_ids"][0].tolist()
        origin_text = ids_to_text(tokenizer, target_ids)

        # Build a fresh MemoryCell, load best memory, then generate from [mem]+<BOS>
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, use_flash_attention_2=args.use_flash_attention_2
        ).to(device)
        memory_dim = get_memory_dim_from_model(base_model)
        mem_cell = MemoryCell(base_model, N_MEM, memory_dim).to(device)
        with torch.no_grad():
            mem_cell.memory.data = torch.from_numpy(best_mem_np).to(mem_cell.memory.data.dtype).to(device)

        # Minimal BOS-only input so generation is driven by memory
        gen_input_ids, gen_attn = build_generation_prompt(tokenizer)
        gen_input_ids = gen_input_ids.to(device)
        gen_attn = gen_attn.to(device)

        # Generate up to target length tokens, greedily
        with torch.no_grad():
            gen_out = mem_cell.generate(
                input_ids=gen_input_ids,
                memory_state=None,
                attention_mask=gen_attn,
                max_new_tokens=len(target_ids),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # HuggingFace's generate returns full sequence (prompt + new tokens).
        # Our prompt length is 1 token (BOS). Slice off that many tokens to get new tokens only.
        if isinstance(gen_out, torch.Tensor):
            full_ids = gen_out[0].tolist()
        else:
            # Some models return BeamSearchDecoderOnlyOutput-like structures.
            full_ids = gen_out.sequences[0].tolist()
        prompt_len = 1  # BOS only
        gen_ids = full_ids[prompt_len:prompt_len + len(target_ids)]
        predicate_text = ids_to_text(tokenizer, gen_ids)

        # Compute token-level match ratio vs target
        match_ratio = compute_match_ratio(target_ids, gen_ids)

        # Collect row
        rows.append({
            "sample_idx": idx,
            "origin_text": origin_text,
            "predicate_text": predicate_text,
            "match_ratio": match_ratio,
            "best_accuracy_train": result.get("best_accuracy", None),
            "original_accuracy_no_mem": result.get("original_accuracy", None),
            "best_loss_train": result.get("best_loss", None),
            "original_loss_no_mem": result.get("original_loss", None),
            "max_length": result.get("max_length", MAX_LEN),
            "n_mem_tokens": result.get("n_mem_tokens", N_MEM),
        })

    # Save CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

    # Also print a short summary
    avg_match = float(np.mean([r["match_ratio"] for r in rows])) if rows else 0.0
    print(f"Saved results to: {out_path}")
    print(f"Average match_ratio over {len(rows)} samples: {avg_match:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()



