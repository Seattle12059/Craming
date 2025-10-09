#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
四种评测模式（可一次性对比）：
1) no_ctx     : 无上下文 QA（最差基线）
2) gold_ctx   : 拼接 gold context 进行 QA（理想上限）
3) mem_token  : 训练 1×mem，推理时 [BOS][mem][prompt] 直接生成（BOS 修复 + inputs_embeds）
4) kv_mem     : 训练 1×mem，推理时先用 [mem] 前向拿 KV，再用 prompt 生成（past_key_values）

输出统一写入 runs/<timestamp_model_n_seed>/ 目录：
- train.log / config.json / preds.csv / preds.jsonl
"""

import argparse
import csv
import json
import re
import string
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
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
    """gold_ctx / no_ctx 用"""
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
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System:\n{SYS_PROMPT}\n\n### User:\n{user}\n\n### Assistant:"

def build_prompt_for_mem(tokenizer, question: str) -> str:
    """mem_token / kv_mem 专用：统一成“复制最短 span”的口径"""
    user = (
        "From the MEMORY tokens prepended above, COPY the shortest exact answer span.\n"
        "If the answer is not present, output exactly: I don't know.\n\n"
        f"Question: {question}\nAnswer (span only):"
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System:\n{SYS_PROMPT}\n\n### User:\n{user}\n\n### Assistant:"

def postprocess_model_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    ans = text.splitlines()[0].strip()
    ans = re.sub(r"^(answer\s*: )\s*", "", ans, flags=re.I)
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
        top_p=1.0, num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )
    gen = out[0][inputs["input_ids"].shape[1]:]  # strip prompt
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return postprocess_model_answer(text)

# =========================
# NON-ORACLE span align
# =========================
def _f1_tokens(pred_tokens, span_tokens):
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
# mem 训练时的上下文截断（避免 OOM）
# =========================
def _extract_keywords(q: str) -> List[str]:
    kws = re.findall(r"[A-Za-z0-9]{3,}", q.lower())
    seen = set(); out = []
    for w in kws:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

def truncate_ctx_for_mem_training(
    question: str,
    context: str,
    tokenizer,
    max_ctx_tokens: int,
    mode: str = "keyword",
    verbose: bool = False,
):
    full_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if full_ids.shape[0] <= max_ctx_tokens:
        return full_ids, None

    if mode == "head":
        s = 0
    else:
        ctx_lc = context.lower(); pos = None
        for kw in _extract_keywords(question):
            j = ctx_lc.find(kw)
            if j != -1: pos = j; break
        if pos is None:
            s = 0
        else:
            prefix = context[:pos]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            start_tok = int(prefix_ids.shape[0])
            s = max(0, min(start_tok - max_ctx_tokens // 3, full_ids.shape[0] - max_ctx_tokens))

    e = s + max_ctx_tokens
    ids_trunc = full_ids[s:e]
    info = f"[mem-trunc] full={full_ids.shape[0]} keep={max_ctx_tokens} range=({s},{e}) mode={mode}"
    if verbose: print(info)
    return ids_trunc.to(tokenizer.device if hasattr(tokenizer, "device") else "cpu"), info

# =========================
# 1×mem Compressor（冻结大模型，只训 mem）
# =========================
class MemCompressor:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        mem_len: int = 1,
        enable_gc: bool = False,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.emb = model.get_input_embeddings()
        hidden = self.emb.embedding_dim
        self.mem = nn.Parameter(torch.zeros(1, mem_len, hidden, device=model.device, dtype=torch.float32))
        nn.init.normal_(self.mem, mean=0.0, std=0.02)
        self.enable_gc = enable_gc
        self._saved_train_flag = None
        self._saved_use_cache = None
        self.logger = logger

    def _log(self, s: str):
        if self.logger is not None: self.logger(s)
        else: print(s)

    def _enter_mem_train_mode(self):
        self._saved_train_flag = self.model.training
        self._saved_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False
        if self.enable_gc and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.train()

    def _exit_mem_train_mode(self):
        self.model.config.use_cache = self._saved_use_cache
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if self._saved_train_flag is False:
            self.model.eval()

    def _build_mem_training_inputs(self, ctx_ids: torch.LongTensor):
        """
        训练：inputs_embeds = [mem] + embed(ctx)
             labels        = [-100] + ctx_ids
        """
        ctx_ids = ctx_ids.to(self.model.device)
        ctx_ids_b = ctx_ids.unsqueeze(0)
        tok_embeds = self.emb(ctx_ids_b)
        mem_cast = self.mem.to(dtype=tok_embeds.dtype)

        inputs_embeds = torch.cat([mem_cast, tok_embeds], dim=1)
        mask_len = self.mem.shape[1]

        labels = torch.cat([
            torch.full((1, mask_len), -100, dtype=torch.long, device=ctx_ids.device),
            ctx_ids_b
        ], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=ctx_ids.device)
        return inputs_embeds, labels, attn

    def train_mem(
        self,
        ctx_ids: torch.LongTensor,
        steps: int = 1000,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.9,
        target_acc: float = 1.0,
        log_every: int = 100,
        verbose: bool = False,
    ):
        # 冻结模型参数，仅优化 mem
        for p in self.model.parameters(): p.requires_grad_(False)
        opt_mem = AdamW([{"params": [self.mem], "lr": lr, "weight_decay": weight_decay, "betas": (beta1, beta2)}])

        best_acc = 0.0
        best_mem = None

        self._enter_mem_train_mode()
        try:
            for step in range(1, steps + 1):
                opt_mem.zero_grad(set_to_none=True)

                inputs_embeds, labels, attn = self._build_mem_training_inputs(ctx_ids)
                outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
                loss = outputs.loss
                loss.backward()

                opt_mem.step()

                with torch.no_grad():
                    acc = self.recon_acc(ctx_ids)

                if acc > best_acc:
                    best_acc = acc
                    best_mem = self.mem.detach().clone()

                if verbose and (step % log_every == 0 or step == 1):
                    self._log(f"[mem-train] step {step}/{steps} loss={loss.item():.4f} recon_acc={acc*100:.2f}%")

                if acc >= target_acc:
                    if verbose:
                        self._log(f"[mem-train] early stop at step {step} recon_acc={acc*100:.2f}%")
                    break
        finally:
            self._exit_mem_train_mode()

        if best_mem is not None:
            with torch.no_grad():
                self.mem.copy_(best_mem)

        return best_acc

    @torch.no_grad()
    def recon_acc(self, ctx_ids: torch.LongTensor) -> float:
        self.model.eval()
        inputs_embeds, labels, attn = self._build_mem_training_inputs(ctx_ids)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn)
        logits = out.logits
        pred = logits[:, :-1, :].argmax(dim=-1)
        tgt = labels[:, 1:]
        mask = (tgt != -100)
        correct = (pred[mask] == tgt[mask]).float().sum().item()
        total = int(mask.sum().item())
        return (correct / total) if total > 0 else 0.0

    # ------- 推理路径 A：mem_token（BOS 修复 + inputs_embeds，不再切片返回） -------
    @torch.inference_mode()
    def answer_with_mem_embeds(self, prompt: str, max_new_tokens=16, temperature=0.0) -> str:
        self.model.eval()
        toks = self.tokenizer(prompt, return_tensors="pt")
        input_ids = toks["input_ids"].to(self.model.device)

        # 保留 BOS 在最前，避免“BOS 被顶后”的分布漂移
        bos_emb = None
        rest_ids = input_ids
        if self.tokenizer.bos_token_id is not None and input_ids.shape[1] >= 1 and int(input_ids[0, 0]) == self.tokenizer.bos_token_id:
            bos_emb = self.emb(input_ids[:, :1])
            rest_ids = input_ids[:, 1:]

        rest_emb = self.emb(rest_ids)
        mem_cast = self.mem.to(dtype=rest_emb.dtype)

        prefix = mem_cast  # 无 AE

        if bos_emb is not None:
            inputs_embeds = torch.cat([bos_emb, prefix, rest_emb], dim=1)
        else:
            inputs_embeds = torch.cat([prefix, rest_emb], dim=1)

        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.model.device)

        gen_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature <= 0 else True,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0, num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
        )
        # 使用 inputs_embeds 时，HF 只返回“新生成”的 tokens，所以无需再切片。
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return postprocess_model_answer(text)

    # ------- 推理路径 B：kv_mem（先 mem 拿 KV，再用 prompt 生成；对生成部分做切片） -------
    @torch.inference_mode()
    def answer_with_mem_kv(self, prompt: str, max_new_tokens=16, temperature=0.0) -> str:
        self.model.eval()
        # 1) 先过 mem 拿 KV
        mem_emb = self.mem.to(self.model.dtype)
        attn_mem = torch.ones((1, mem_emb.shape[1]), dtype=torch.long, device=self.model.device)
        out = self.model(inputs_embeds=mem_emb, attention_mask=attn_mem, use_cache=True)
        past = out.past_key_values

        # 2) 再正常喂 prompt ids
        toks = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in toks.items()}
        prompt_len = inputs["input_ids"].shape[1]
        gen = self.model.generate(
            **inputs,
            past_key_values=past,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature <= 0 else True,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0, num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
        )
        new_ids = gen[0][prompt_len:]  # 只取新增部分
        gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return postprocess_model_answer(gen_text)

# =========================
# 日志器
# =========================
class SimpleLogger:
    def __init__(self, log_path: Path):
        self.f = open(log_path, "w", encoding="utf-8")
    def __call__(self, s: str):
        print(s); self.f.write(s + "\n"); self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv_out", type=str, default="preds.csv")
    ap.add_argument("--save_preds", type=str, default="preds.jsonl")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","auto"])
    ap.add_argument("--verbose", action="store_true")

    # mem 超参
    ap.add_argument("--mem_len", type=int, default=1)
    ap.add_argument("--mem_steps", type=int, default=1000)
    ap.add_argument("--mem_lr", type=float, default=1e-2)
    ap.add_argument("--mem_wd", type=float, default=1e-2)
    ap.add_argument("--mem_beta1", type=float, default=0.9)
    ap.add_argument("--mem_beta2", type=float, default=0.9)
    ap.add_argument("--mem_target_acc", type=float, default=1.0)
    ap.add_argument("--mem_log_every", type=int, default=100)

    # mem 训练专用截断
    ap.add_argument("--max_ctx_tokens", type=int, default=1024)
    ap.add_argument("--ctx_window", type=str, default="keyword", choices=["keyword","head"])
    ap.add_argument("--enable_gc", action="store_true")

    # 结果目录
    ap.add_argument("--out_dir_base", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default=None)

    # 评测模式选择
    ap.add_argument("--modes", type=str, default="all",
                    help="逗号分隔: no_ctx,gold_ctx,mem_token,kv_mem 或 all")

    args = ap.parse_args()

    # ========== 结果目录 ==========
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", Path(args.model).name)
    run_name = args.run_name or f"{ts}_{model_slug}_n{args.n}_seed{args.seed}"
    run_dir = Path(args.out_dir_base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / Path(args.csv_out).name
    jsonl_path = run_dir / Path(args.save_preds).name if args.save_preds else None
    log_path = run_dir / "train.log"
    cfg_path = run_dir / "config.json"

    logger = SimpleLogger(log_path)
    logger(f"[run-dir] {run_dir}")

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    random.seed(args.seed); torch.manual_seed(args.seed)

    split_expr = f"{args.split}[:{args.n}]"
    ds = load_dataset("LLukas22/nq-simplified", split=split_expr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda" if args.device == "cuda" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 解析模式
    if args.modes.strip().lower() == "all":
        modes = ["no_ctx","gold_ctx","mem_token","kv_mem"]
    else:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # 统计器
    agg = {m: {"em":0.0,"f1":0.0,"n":0} for m in ["no_ctx","gold_ctx","mem_token","kv_mem"]}

    header = [
        "context_len_tokens","id","question","context","gold_answers",
        # 4 模式预测与指标
        "pred_no_ctx","em_no_ctx","f1_no_ctx",
        "pred_gold_ctx","em_gold_ctx","f1_gold_ctx",
        "pred_mem_token","em_mem_token","f1_mem_token",
        "pred_kv_mem","em_kv_mem","f1_kv_mem",
        # mem 训练信息
        "mem_recon_acc","mem_trunc_info",
        # 运行配置
        "mem_len"
    ]

    rows = []
    jsonl_rows = [] if jsonl_path is not None else None

    for i, ex in enumerate(ds):
        q = ex["question"]; ctx = ex["context"]
        golds = ex.get("answers", {}).get("text", [])
        gold_str = " || ".join(golds)

        # 原始 ctx 长度（报告用）
        ctx_ids_full = tokenizer(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        ctx_len_tokens_full = int(ctx_ids_full.shape[0])

        # (1) no_ctx
        pred_no_ctx = em_no_ctx = f1_no_ctx = ""
        if "no_ctx" in modes:
            prompt1 = build_prompt(tokenizer, q, context=None)
            pred1 = generate_answer(model, tokenizer, prompt1, args.max_new_tokens, args.temperature)
            em1, f11 = metric_em_f1(pred1, golds)
            agg["no_ctx"]["em"] += em1; agg["no_ctx"]["f1"] += f11; agg["no_ctx"]["n"] += 1
            pred_no_ctx, em_no_ctx, f1_no_ctx = pred1, f"{em1:.3f}", f"{f11:.3f}"

        # (2) gold_ctx
        pred_gold_ctx = em_gold_ctx = f1_gold_ctx = ""
        if "gold_ctx" in modes:
            prompt2 = build_prompt(tokenizer, q, context=ctx)
            pred2_raw = generate_answer(model, tokenizer, prompt2, args.max_new_tokens, args.temperature)
            pred2_span = spanize_to_context(pred2_raw, ctx)
            em2, f12 = metric_em_f1(pred2_span, golds)
            agg["gold_ctx"]["em"] += em2; agg["gold_ctx"]["f1"] += f12; agg["gold_ctx"]["n"] += 1
            pred_gold_ctx, em_gold_ctx, f1_gold_ctx = pred2_span, f"{em2:.3f}", f"{f12:.3f}"

        # mem 相关（3、4）
        pred_mem_token = em_mem_token = f1_mem_token = ""
        pred_kv_mem = em_kv_mem = f1_kv_mem = ""
        recon_best = 0.0; trunc_info = None

        if ("mem_token" in modes) or ("kv_mem" in modes):
            # 训练 mem（只对 mem 训练使用截断）
            ctx_ids_for_mem, trunc_info = truncate_ctx_for_mem_training(
                q, ctx, tokenizer, max_ctx_tokens=args.max_ctx_tokens, mode=args.ctx_window, verbose=args.verbose
            )
            ctx_ids_for_mem = ctx_ids_for_mem.to(model.device)

            compressor = MemCompressor(
                model, tokenizer,
                mem_len=args.mem_len, enable_gc=args.enable_gc,
                logger=logger if args.verbose else None,
            )
            recon_best = compressor.train_mem(
                ctx_ids=ctx_ids_for_mem,
                steps=args.mem_steps,
                lr=args.mem_lr,
                weight_decay=args.mem_wd,
                beta1=args.mem_beta1,
                beta2=args.mem_beta2,
                target_acc=args.mem_target_acc,
                log_every=args.mem_log_every,
                verbose=args.verbose,
            )

            # (3) mem_token
            if "mem_token" in modes:
                prompt_mem = build_prompt_for_mem(tokenizer, q)
                pred_m = compressor.answer_with_mem_embeds(
                    prompt_mem, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
                em3, f13 = metric_em_f1(pred_m, golds)
                agg["mem_token"]["em"] += em3; agg["mem_token"]["f1"] += f13; agg["mem_token"]["n"] += 1
                pred_mem_token, em_mem_token, f1_mem_token = pred_m, f"{em3:.3f}", f"{f13:.3f}"

            # (4) kv_mem
            if "kv_mem" in modes:
                prompt_kv = build_prompt_for_mem(tokenizer, q)
                pred_kv = compressor.answer_with_mem_kv(
                    prompt_kv, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
                em4, f14 = metric_em_f1(pred_kv, golds)
                agg["kv_mem"]["em"] += em4; agg["kv_mem"]["f1"] += f14; agg["kv_mem"]["n"] += 1
                pred_kv_mem, em_kv_mem, f1_kv_mem = pred_kv, f"{em4:.3f}", f"{f14:.3f}"

            # 释放
            del compressor
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        rows.append([
            ctx_len_tokens_full, i, q, ctx, gold_str,
            pred_no_ctx, em_no_ctx, f1_no_ctx,
            pred_gold_ctx, em_gold_ctx, f1_gold_ctx,
            pred_mem_token, em_mem_token, f1_mem_token,
            pred_kv_mem, em_kv_mem, f1_kv_mem,
            f"{recon_best:.3f}", trunc_info,
            args.mem_len,
        ])

        if jsonl_rows is not None:
            jsonl_rows.append({
                "id": i, "question": q, "context": ctx, "gold_answers": golds,
                "pred_no_ctx": pred_no_ctx, "pred_gold_ctx": pred_gold_ctx,
                "pred_mem_token": pred_mem_token, "pred_kv_mem": pred_kv_mem,
                "em_no_ctx": float(em_no_ctx) if em_no_ctx else None,
                "f1_no_ctx": float(f1_no_ctx) if f1_no_ctx else None,
                "em_gold_ctx": float(em_gold_ctx) if em_gold_ctx else None,
                "f1_gold_ctx": float(f1_gold_ctx) if f1_gold_ctx else None,
                "em_mem_token": float(em_mem_token) if em_mem_token else None,
                "f1_mem_token": float(f1_mem_token) if f1_mem_token else None,
                "em_kv_mem": float(em_kv_mem) if em_kv_mem else None,
                "f1_kv_mem": float(f1_kv_mem) if f1_kv_mem else None,
                "context_len_tokens": ctx_len_tokens_full,
                "mem_recon_acc": recon_best,
                "mem_trunc_info": trunc_info,
                "mem_len": args.mem_len,
            })

        if args.verbose and (i+1) % 10 == 0:
            logger(f"[{i+1}/{args.n}] processed.")

    logger("\n=== Final Averages ===")
    for m in modes:
        n = max(1, agg[m]["n"])
        em = agg[m]["em"]/n
        f1 = agg[m]["f1"]/n
        logger(f"{m:>9} : EM={em*100:.1f}  F1={f1*100:.1f}  (N={n})")

    logger(f"\nSaving CSV -> {csv_path}")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)

    if jsonl_rows is not None:
        jsonl_path = Path(jsonl_path)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in jsonl_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger(f"Saved JSONL -> {jsonl_path}")

    logger.close()

if __name__ == "__main__":
    main()
