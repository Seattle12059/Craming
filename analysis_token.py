#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py — n→1 token 压缩效率综合分析（参考你稳定不 OOM 的实现做了显存友好改造）

关键改动：
  - 默认单卡加载（device_map="cuda"），避免 auto 分片导致的 meta/cpu offload 碎片。
  - 可选 SDPA/Flash-Attn2 注意力实现（--attn_impl），更省显存。
  - 可选 4-bit 量化（--load_in_4bit，需要 bitsandbytes）。
  - mem 训练阶段自动关闭 KV cache 并开启 gradient checkpointing（use_reentrant=False）。
  - 仅在 mem 训练时对 context 做截断；报告里同时记录 full 与 used。
  - 使用 from_pretrained(dtype=...)（兼容 "torch_dtype is deprecated" 提示）。

输出：
  - 明细 CSV（每条样本的 n_full / n_used / steps / seconds / tok/s 等）
  - 汇总 JSON（均值、分位数、相关性、分桶）
  - Markdown 报告（可读）
"""

import argparse
import csv
import json
import math
import os
import random
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 可选：bitsandbytes 4bit
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# =========================
# 工具函数（截断、统计）
# =========================

def _extract_keywords(q: str) -> List[str]:
    kws = re.findall(r"[A-Za-z0-9]{3,}", (q or "").lower())
    seen, out = set(), []
    for w in kws:
        if w not in seen:
            seen.add(w); out.append(w)
    return out


def truncate_ctx_for_mem_training(
    question: str,
    context: str,
    tokenizer: AutoTokenizer,
    max_ctx_tokens: int,
    mode: str = "keyword",
) -> Tuple[torch.LongTensor, str, int]:
    """仅用于 mem 训练阶段的上下文截断（避免 OOM）。"""
    context = context or ""
    full_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if full_ids.shape[0] <= max_ctx_tokens:
        return full_ids, f"full<=limit keep={full_ids.shape[0]}", int(full_ids.shape[0])

    if mode == "head":
        s = 0
    else:
        ctx_lc, pos = context.lower(), None
        for kw in _extract_keywords(question or ""):
            j = ctx_lc.find(kw)
            if j != -1:
                pos = j; break
        if pos is None:
            s = 0
        else:
            prefix = context[:pos]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            start_tok = int(prefix_ids.shape[0])
            s = max(0, min(start_tok - max_ctx_tokens // 3, full_ids.shape[0] - max_ctx_tokens))

    e = s + max_ctx_tokens
    ids_trunc = full_ids[s:e]
    info = f"range=({s},{e}) keep={ids_trunc.shape[0]} mode={mode}"
    return ids_trunc, info, int(ids_trunc.shape[0])


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


def percentiles(arr: List[float], ps=(50, 75, 90, 95, 99)) -> Dict[str, float]:
    if not arr:
        return {f"p{p}": 0.0 for p in ps}
    data = np.array(arr, dtype=float)
    return {f"p{p}": float(np.percentile(data, p)) for p in ps}


def corrcoef(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    X, Y = np.array(x, dtype=float), np.array(y, dtype=float)
    if X.std() == 0 or Y.std() == 0:
        return 0.0
    return float(np.corrcoef(X, Y)[0, 1])


# =========================
# 1×mem 压缩器（显存友好）
# =========================

class MemCompressor(nn.Module):
    """冻结大模型，仅优化 1×mem 参数；前向拼接前把 mem cast 成模型嵌入 dtype。
    训练目标：teacher-forcing 下重建 context（next-token 交叉熵）。
    """
    def __init__(self, model: AutoModelForCausalLM, enable_gc: bool = False):
        super().__init__()
        self.model = model
        self.emb = model.get_input_embeddings()
        H = self.emb.embedding_dim
        self.mem = nn.Parameter(torch.zeros(1, 1, H, device=model.device, dtype=torch.float32))
        nn.init.normal_(self.mem, mean=0.0, std=0.02)
        self.enable_gc = enable_gc
        self._saved_flag = None
        self._saved_cache = None

    def _enter_train_mode(self):
        self._saved_flag = self.model.training
        self._saved_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False
        if self.enable_gc and hasattr(self.model, "gradient_checkpointing_enable"):
            # use_reentrant=False 更稳且显存更省
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        self.model.train()

    def _exit_train_mode(self):
        self.model.config.use_cache = self._saved_cache
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if self._saved_flag is False:
            self.model.eval()

    def _build_inputs(self, ctx_ids: torch.LongTensor):
        ctx_ids = ctx_ids.to(self.model.device)
        ctx_ids_b = ctx_ids.unsqueeze(0)
        tok_embeds = self.emb(ctx_ids_b)           # [1, T, H]
        mem_cast = self.mem.to(dtype=tok_embeds.dtype)
        inputs_embeds = torch.cat([mem_cast, tok_embeds], dim=1)
        labels = torch.cat([
            torch.full((1, 1), -100, dtype=torch.long, device=ctx_ids.device),
            ctx_ids_b,
        ], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=ctx_ids.device)
        return inputs_embeds, labels, attn

    @torch.no_grad()
    def recon_acc(self, ctx_ids: torch.LongTensor) -> float:
        self.model.eval()
        inputs_embeds, labels, attn = self._build_inputs(ctx_ids)
        logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attn).logits
        pred = logits[:, :-1, :].argmax(dim=-1)
        tgt  = labels[:, 1:]
        mask = (tgt != -100)
        total = int(mask.sum().item())
        if total == 0:
            return 0.0
        correct = (pred[mask] == tgt[mask]).float().sum().item()
        return float(correct / total)

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
    ) -> Dict[str, float]:
        # 冻结大模型，仅优化 mem
        for p in self.model.parameters():
            p.requires_grad_(False)
        opt = AdamW([{"params": [self.mem], "lr": lr, "weight_decay": weight_decay, "betas": (beta1, beta2)}])

        best_acc = -1.0
        best_mem = None
        steps_to_target = None
        t0 = time.perf_counter()

        self._enter_train_mode()
        try:
            for step in range(1, steps + 1):
                opt.zero_grad(set_to_none=True)
                inputs_embeds, labels, attn = self._build_inputs(ctx_ids)
                out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
                loss = out.loss
                loss.backward()
                opt.step()

                with torch.no_grad():
                    acc = self.recon_acc(ctx_ids)

                if acc > best_acc:
                    best_acc = acc
                    best_mem = self.mem.detach().clone()

                if steps_to_target is None and acc >= target_acc:
                    steps_to_target = step
                    if verbose:
                        print(f"[mem] reach target={target_acc:.3f} at step {step}, acc={acc:.4f}")
                    break

                if verbose and (step % log_every == 0 or step == 1):
                    print(f"[mem] step {step}/{steps} loss={loss.item():.4f} acc={acc:.4f}")
        finally:
            self._exit_train_mode()

        t1 = time.perf_counter()
        if best_mem is not None:
            with torch.no_grad():
                self.mem.copy_(best_mem)

        return {
            "best_acc": float(best_acc if best_acc >= 0 else 0.0),
            "steps_run": int(steps_to_target if steps_to_target is not None else steps),
            "steps_to_target": int(steps_to_target) if steps_to_target is not None else None,
            "early_stop": bool(steps_to_target is not None),
            "seconds": float(t1 - t0),
        }


# =========================
# 汇总/导出
# =========================

@dataclass
class ExampleStat:
    idx: int
    ctx_len_full: int
    ctx_len_used: int
    steps_run: int
    steps_to_target: Optional[int]
    early_stop: bool
    best_acc: float
    seconds: float
    tokens_processed: int
    tokens_per_sec: float
    trunc_info: str


def summarize(stats: List[ExampleStat]) -> Dict:
    n = len(stats)
    if n == 0:
        return {}

    seconds = [s.seconds for s in stats]
    steps   = [s.steps_run for s in stats]
    accs    = [s.best_acc for s in stats]
    n_full  = [s.ctx_len_full for s in stats]
    n_used  = [s.ctx_len_used for s in stats]
    tps     = [s.tokens_per_sec for s in stats]
    tokens  = [s.tokens_processed for s in stats]

    def avg(a):
        return float(np.mean(a)) if a else 0.0

    summary = {
        "num_examples": n,
        "seconds_avg": avg(seconds),
        "seconds_sum": float(np.sum(seconds)),
        "steps_avg": avg(steps),
        "acc_avg": avg(accs),
        "ctx_len_full_avg": avg(n_full),
        "ctx_len_used_avg": avg(n_used),
        "tokens_processed_sum": int(np.sum(tokens)),
        "tokens_per_sec_avg": avg(tps),
        "seconds_percentiles": percentiles(seconds),
        "steps_percentiles": percentiles(steps),
        "acc_percentiles": percentiles(accs),
        "corr_ctxfull_steps": corrcoef(n_full, steps),
        "corr_ctxfull_seconds": corrcoef(n_full, seconds),
        "corr_ctxused_steps": corrcoef(n_used, steps),
        "corr_ctxused_seconds": corrcoef(n_used, seconds),
    }

    # 分桶（按 full ctx len）
    edges = [256, 512, 1024, 2048, 4096, 8192]
    def bucketize(vs: List[int], edges: List[int]):
        out = []
        for v in vs:
            i = 0
            while i < len(edges) and v > edges[i]:
                i += 1
            out.append(i)
        return out

    bucket_idx = bucketize(n_full, edges)
    buckets = {}
    for bi in range(len(edges) + 1):
        mask = [i for i, b in enumerate(bucket_idx) if b == bi]
        if not mask:
            continue
        rng = (
            f"<= {edges[0]}" if bi == 0 else
            f"{edges[bi-1]+1}–{edges[bi]}" if bi < len(edges) else f"> {edges[-1]}"
        )
        buckets[rng] = {
            "count": len(mask),
            "seconds_avg": avg([seconds[i] for i in mask]),
            "steps_avg": avg([steps[i] for i in mask]),
            "acc_avg": avg([accs[i] for i in mask]),
            "tokens_per_sec_avg": avg([tps[i] for i in mask]),
        }
    summary["buckets_by_ctx_full"] = buckets
    return summary


def write_csv(path: str, stats: List[ExampleStat]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "id","ctx_len_full","ctx_len_used","steps_run","steps_to_target","early_stop",
        "best_acc","seconds","tokens_processed","tokens_per_sec","trunc_info",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in stats:
            w.writerow([
                s.idx, s.ctx_len_full, s.ctx_len_used, s.steps_run,
                s.steps_to_target if s.steps_to_target is not None else "",
                int(s.early_stop), f"{s.best_acc:.6f}", f"{s.seconds:.6f}",
                s.tokens_processed, f"{s.tokens_per_sec:.3f}", s.trunc_info,
            ])


def write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_report_md(path: str, args, summary: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# n→1 token 压缩效率报告\n")
        f.write("## 运行参数\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"- **{k}**: {v}\n")
        f.write("## 汇总指标\n")
        if not summary:
            f.write("无数据\n"); return
        f.write(
            (
                "- 样本数: **{num_examples}**\n"
                "- 耗时（总/均值）: **{seconds_sum:.2f}s** / **{seconds_avg:.2f}s**\n"
                "- 步数（均值）: **{steps_avg:.1f}**\n"
                "- 重建准确率（均值）: **{acc_avg:.4f}**\n"
                "- 上下文长度（均值，full/used）: **{ctx_len_full_avg:.1f}** / **{ctx_len_used_avg:.1f}**\n"
                "- 吞吐（均值）: **{tokens_per_sec_avg:.1f} tok/s**\n"
            ).format(**summary)
        )
        f.write("### 分位数\n")
        sp = summary["seconds_percentiles"]
        stp = summary["steps_percentiles"]
        f.write(f"- 耗时 p50/p90/p95: {sp.get('p50',0.0):.2f}s / {sp.get('p90',0.0):.2f}s / {sp.get('p95',0.0):.2f}s\n")
        f.write(f"- 步数 p50/p90/p95: {stp.get('p50',0.0):.0f} / {stp.get('p90',0.0):.0f} / {stp.get('p95',0.0):.0f}\n")
        f.write("### 相关性（Pearson r）\n")
        f.write(
            f"- ctx_full ↔ steps: {summary.get('corr_ctxfull_steps',0.0):.3f}\n"
            f"- ctx_full ↔ seconds: {summary.get('corr_ctxfull_seconds',0.0):.3f}\n"
            f"- ctx_used ↔ steps: {summary.get('corr_ctxused_steps',0.0):.3f}\n"
            f"- ctx_used ↔ seconds: {summary.get('corr_ctxused_seconds',0.0):.3f}\n"
        )
        f.write("### 按上下文长度分桶\n\n")
        buckets = summary.get("buckets_by_ctx_full", {})
        if not buckets:
            f.write("(无)\n")
        else:
            f.write("区间 | 样本数 | 平均步数 | 平均耗时(s) | 平均准确率 | 吞吐(tok/s)\n")
            f.write("---|---:|---:|---:|---:|---:\n")
            for rng, b in buckets.items():
                f.write(f"{rng} | {b['count']} | {b['steps_avg']:.1f} | {b['seconds_avg']:.2f} | {b['acc_avg']:.4f} | {b['tokens_per_sec_avg']:.1f}\n")


# =========================
# 主流程
# =========================

def main():
    import torch  # 添加这行
    ap = argparse.ArgumentParser()
    # 数据/模型
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="LLukas22/nq-simplified")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--question_field", type=str, default="question")
    ap.add_argument("--context_field", type=str, default="context")

    # 设备/加载
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "auto"])  # 默认单卡
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa","flash_attention_2","eager"], help="注意力后端")
    ap.add_argument("--load_in_4bit", action="store_true", help="使用 bitsandbytes 4bit 量化")

    # 随机数
    ap.add_argument("--seed", type=int, default=42)

    # mem 超参
    ap.add_argument("--mem_steps", type=int, default=1000)
    ap.add_argument("--mem_lr", type=float, default=1e-2)
    ap.add_argument("--mem_wd", type=float, default=1e-2)
    ap.add_argument("--mem_beta1", type=float, default=0.9)
    ap.add_argument("--mem_beta2", type=float, default=0.9)
    ap.add_argument("--mem_target_acc", type=float, default=1.0)
    ap.add_argument("--mem_log_every", type=int, default=100)
    ap.add_argument("--enable_gc", action="store_true")

    # 仅对 mem 训练生效的上下文截断
    ap.add_argument("--max_ctx_tokens", type=int, default=512)
    ap.add_argument("--ctx_window", type=str, default="keyword", choices=["keyword","head"])

    # 输出
    ap.add_argument("--csv_out", type=str, default="mem_efficiency_stats.csv")
    ap.add_argument("--json_out", type=str, default="mem_efficiency_summary.json")
    ap.add_argument("--report_out", type=str, default="mem_report.md")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 数据
    split_expr = f"{args.split}[:{args.n}]"
    ds = load_dataset(args.dataset, split=split_expr)

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 模型加载（显存友好）
    load_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation=args.attn_impl)

    if args.load_in_4bit and _HAS_BNB:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        load_kwargs.update({
            "quantization_config": bnb_cfg,
            "device_map": "auto",   # 量化通常配合 auto 更稳
        })
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "cuda" if args.device == "cuda" else "auto"
        load_kwargs.update({"dtype": dtype, "device_map": device_map})

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    # 进一步启用 SDPA 的高效内核（若可用）
    try:
        import torch.backends.cuda
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

    stats: List[ExampleStat] = []

    for i, ex in enumerate(ds):
        q = ex.get(args.question_field, "")
        ctx = ex.get(args.context_field, "")

        # full token 长度（报告用）
        ctx_ids_full = tokenizer(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        n_full = int(ctx_ids_full.shape[0])

        # 仅 mem 训练截断
        ctx_ids_used, trunc_info, n_used = truncate_ctx_for_mem_training(
            q, ctx, tokenizer, max_ctx_tokens=args.max_ctx_tokens, mode=args.ctx_window
        )
        ctx_ids_used = ctx_ids_used.to(model.device)

        compressor = MemCompressor(model, enable_gc=args.enable_gc)
        res = compressor.train_mem(
            ctx_ids=ctx_ids_used,
            steps=args.mem_steps,
            lr=args.mem_lr,
            weight_decay=args.mem_wd,
            beta1=args.mem_beta1,
            beta2=args.mem_beta2,
            target_acc=args.mem_target_acc,
            log_every=args.mem_log_every,
            verbose=args.verbose,
        )

        steps_run = int(res["steps_run"])  # 达标即停
        seconds   = float(res["seconds"]) if not math.isnan(res["seconds"]) else 0.0
        tokens_processed = int(n_used * steps_run)
        tps = _safe_div(tokens_processed, seconds)

        stats.append(
            ExampleStat(
                idx=i,
                ctx_len_full=n_full,
                ctx_len_used=n_used,
                steps_run=steps_run,
                steps_to_target=res["steps_to_target"],
                early_stop=bool(res["early_stop"]),
                best_acc=float(res["best_acc"]),
                seconds=seconds,
                tokens_processed=tokens_processed,
                tokens_per_sec=tps,
                trunc_info=trunc_info,
            )
        )

        del compressor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.verbose and (i + 1) % 10 == 0:
            print(f"[{i+1}/{args.n}] n_full={n_full} n_used={n_used} steps={steps_run} time={seconds:.2f}s acc={res['best_acc']:.4f}")

    # 导出
    write_csv(args.csv_out, stats)
    summary = summarize(stats)
    write_json(args.json_out, summary)
    write_report_md(args.report_out, args, summary)

    print("=== n→1 token 压缩效率（摘要） ===")
    if summary:
        print(
            "样本数={num_examples}  平均步数={steps_avg:.1f}  平均耗时={seconds_avg:.2f}s  平均吞吐={tokens_per_sec_avg:.1f} tok/s".format(**summary)
        )
        sp = summary["seconds_percentiles"]; stp = summary["steps_percentiles"]
        print(
            "耗时 p50/p90/p95: {p50:.2f}s / {p90:.2f}s / {p95:.2f}s;  步数 p50/p90/p95: {p50s:.0f}/{p90s:.0f}/{p95s:.0f}".format(
                p50=sp.get("p50",0.0), p90=sp.get("p90",0.0), p95=sp.get("p95",0.0),
                p50s=stp.get("p50",0.0), p90s=stp.get("p90",0.0), p95s=stp.get("p95",0.0),
            )
        )
        print(f"CSV -> {args.csv_out}")
        print(f"JSON -> {args.json_out}")
        print(f"Report -> {args.report_out}")
    else:
        print("无可用数据。")


if __name__ == "__main__":
    main()
