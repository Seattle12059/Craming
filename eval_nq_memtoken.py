#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, re, string, random
from typing import List, Tuple

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
    s = re.sub(r'^[\"\'`\-–—\(\[]+|[\"\'`\-–—\)\]]+$', "", s)
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
# 仅用于 mem 训练的上下文截断（避免 OOM）
# =========================
def _extract_keywords(q: str) -> List[str]:
    # 取长度>=3的字母数字词，去重
    kws = re.findall(r"[A-Za-z0-9]{3,}", q.lower())
    seen = set()
    out = []
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
    verbose: bool = False
):
    """
    只在 mem 训练阶段使用的截断策略：
      - 'keyword': 用问题中的关键词在原文里找第一次出现的位置，窗口居中于其附近
      - 'head'    : 直接取开头 max_ctx_tokens
    返回: (ctx_ids_trunc, info_str)
    """
    full_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if full_ids.shape[0] <= max_ctx_tokens:
        return full_ids, None

    if mode == "head":
        s = 0
    else:
        # keyword window
        ctx_lc = context.lower()
        pos = None
        for kw in _extract_keywords(question):
            j = ctx_lc.find(kw)
            if j != -1:
                pos = j
                break
        if pos is None:
            s = 0
        else:
            # 估算关键字处的 token 起点：对前缀字符做一次分词计数
            prefix = context[:pos]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            start_tok = int(prefix_ids.shape[0])
            # 让关键字大致落在窗口的 1/3 处，给后文留空间
            s = max(0, min(start_tok - max_ctx_tokens // 3, full_ids.shape[0] - max_ctx_tokens))

    e = s + max_ctx_tokens
    ids_trunc = full_ids[s:e]
    info = f"[mem-trunc] full={full_ids.shape[0]}  keep={max_ctx_tokens}  range=({s},{e}) mode={mode}"
    if verbose:
        print(info)
    return ids_trunc.to(tokenizer.device if hasattr(tokenizer, "device") else "cpu"), info

# =========================
# ③ 单样本 1×mem token 压缩/重建/QA
# =========================
class MemCompressor:
    """
    冻结模型，仅优化一个 FP32 的 mem 参数；前向拼接前把 mem cast 成模型嵌入 dtype（fp16/bf16）。
    训练目标：teacher forcing 下重建 context。训练后用 mem + 问题做 QA。
    支持在训练开始/结束时切换 gradient checkpointing 及 use_cache 以省显存。
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, mem_len: int = 1, enable_gc: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.emb = model.get_input_embeddings()
        hidden = self.emb.embedding_dim
        self.mem = nn.Parameter(torch.zeros(1, mem_len, hidden, device=model.device, dtype=torch.float32))
        nn.init.normal_(self.mem, mean=0.0, std=0.02)
        self.enable_gc = enable_gc
        self._saved_train_flag = None
        self._saved_use_cache = None

    def _enter_mem_train_mode(self):
        # 启用梯度检查点并关闭缓存以降低显存
        self._saved_train_flag = self.model.training
        self._saved_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False
        if self.enable_gc and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.train()  # LLaMA 默认无 dropout，训练/评估差异可忽略；需要训练态以启用 checkpointing

    def _exit_mem_train_mode(self):
        # 还原
        self.model.config.use_cache = self._saved_use_cache
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if self._saved_train_flag is False:
            self.model.eval()

    def _build_mem_training_inputs(self, ctx_ids: torch.LongTensor):
        """
        inputs_embeds = [mem] + embed(ctx_ids)
        labels        = [-100] + ctx_ids
        """
        ctx_ids = ctx_ids.to(self.model.device)
        T = ctx_ids.shape[0]
        ctx_ids_b = ctx_ids.unsqueeze(0)                # [1, T]
        tok_embeds = self.emb(ctx_ids_b)                # [1, T, H], model dtype
        mem_cast = self.mem.to(dtype=tok_embeds.dtype)  # cast
        inputs_embeds = torch.cat([mem_cast, tok_embeds], dim=1)  # [1, 1+T, H]
        labels = torch.cat(
            [torch.full((1, 1), -100, dtype=torch.long, device=ctx_ids.device), ctx_ids_b],
            dim=1
        )
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=ctx_ids.device)
        return inputs_embeds, labels, attn

    def train_mem(self,
                  ctx_ids: torch.LongTensor,
                  steps: int = 1000,
                  lr: float = 1e-2,
                  weight_decay: float = 1e-2,
                  beta1: float = 0.9,
                  beta2: float = 0.9,
                  target_acc: float = 1.0,
                  log_every: int = 100,
                  verbose: bool = False):
        # 冻结模型参数，仅优化 mem
        for p in self.model.parameters(): p.requires_grad_(False)

        opt = AdamW([{"params": [self.mem], "lr": lr, "weight_decay": weight_decay, "betas": (beta1, beta2)}])

        best_acc = 0.0
        best_mem = None

        self._enter_mem_train_mode()
        try:
            for step in range(1, steps + 1):
                opt.zero_grad(set_to_none=True)
                inputs_embeds, labels, attn = self._build_mem_training_inputs(ctx_ids)
                outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
                loss = outputs.loss
                loss.backward()
                opt.step()

                with torch.no_grad():
                    acc = self.recon_acc(ctx_ids)

                if acc > best_acc:
                    best_acc = acc
                    best_mem = self.mem.detach().clone()

                if verbose and (step % log_every == 0 or step == 1):
                    print(f"[mem-train] step {step}/{steps}  loss={loss.item():.4f}  recon_acc={acc*100:.2f}%")

                if acc >= target_acc:
                    if verbose:
                        print(f"[mem-train] early stop at step {step} with recon_acc={acc*100:.2f}%")
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

    @torch.inference_mode()
    def answer_with_mem(self, prompt: str, max_new_tokens=16, temperature=0.0) -> str:
        self.model.eval()
        toks = self.tokenizer(prompt, return_tensors="pt")
        input_ids = toks["input_ids"].to(self.model.device)
        prompt_embeds = self.emb(input_ids)                 # [1, L, H]
        mem_cast = self.mem.to(dtype=prompt_embeds.dtype)   # cast
        inputs_embeds = torch.cat([mem_cast, prompt_embeds], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.model.device)

        gen_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature <= 0 else True,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
        )
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return postprocess_model_answer(text)

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
    ap.add_argument("--csv_out", type=str, default="preds_nq_llama8b_mem1.csv")
    ap.add_argument("--save_preds", type=str, default=None)
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

    # NEW: 仅对 mem 训练生效的上下文截断参数
    ap.add_argument("--max_ctx_tokens", type=int, default=1024, help="Upper bound of context tokens used ONLY for mem training to avoid OOM")
    ap.add_argument("--ctx_window", type=str, default="keyword", choices=["keyword","head"], help="Keyword-centered window or take head")
    ap.add_argument("--enable_gc", action="store_true", help="Enable gradient checkpointing during mem training")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    header = [
        "context_len_tokens","id","question","context","gold_answers",
        "pred_no_ctx","pred_with_ctx","em_no_ctx","f1_no_ctx","em_with_ctx","f1_with_ctx",
        "1mem token 推理结果","还原正确率",
    ]

    em_1 = f1_1 = em_2 = f1_2 = 0.0
    mem_em = mem_f1 = 0.0
    rows, jsonl = [], ([] if args.save_preds else None)

    for i, ex in enumerate(ds):
        q = ex["question"]
        ctx = ex["context"]
        golds = ex.get("answers", {}).get("text", [])
        gold_str = " || ".join(golds)

        # === (A) 统计原始 context token 长度（仅用于报告，不用于 mem 训练的长度） ===
        ctx_ids_full = tokenizer(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        ctx_len_tokens_full = int(ctx_ids_full.shape[0])

        # (1) no context
        prompt1 = build_prompt(tokenizer, q, context=None)
        pred1 = generate_answer(model, tokenizer, prompt1, args.max_new_tokens, args.temperature)
        em1, f11 = metric_em_f1(pred1, golds)

        # (2) with full context（评测时仍用完整上下文，不截断）
        prompt2 = build_prompt(tokenizer, q, context=ctx)
        pred2_raw = generate_answer(model, tokenizer, prompt2, args.max_new_tokens, args.temperature)
        pred2_span = spanize_to_context(pred2_raw, ctx)
        em2, f12 = metric_em_f1(pred2_span, golds)

        # (3) mem token：仅在训练 mem 时对 context 截断，避免 OOM
        ctx_ids_for_mem, info = truncate_ctx_for_mem_training(
            q, ctx, tokenizer, max_ctx_tokens=args.max_ctx_tokens, mode=args.ctx_window, verbose=args.verbose
        )
        ctx_ids_for_mem = ctx_ids_for_mem.to(model.device)

        compressor = MemCompressor(model, tokenizer, mem_len=args.mem_len, enable_gc=args.enable_gc)
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
        prompt_memqa = build_prompt(tokenizer, q, context=None)
        pred_mem = compressor.answer_with_mem(prompt_memqa, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        em3, f13 = metric_em_f1(pred_mem, golds)
        mem_em += em3; mem_f1 += f13

        em_1 += em1; f1_1 += f11; em_2 += em2; f1_2 += f12

        rows.append([
            ctx_len_tokens_full, i, q, ctx, gold_str,
            pred1, pred2_span,
            f"{em1:.3f}", f"{f11:.3f}", f"{em2:.3f}", f"{f12:.3f}",
            pred_mem, f"{recon_best:.3f}",
        ])

        if jsonl is not None:
            jsonl.append({
                "id": i,
                "question": q,
                "context": ctx,
                "gold_answers": golds,
                "pred_no_ctx": pred1,
                "pred_with_ctx": pred2_span,
                "pred_with_1mem": pred_mem,
                "em_no_ctx": em1, "f1_no_ctx": f11,
                "em_with_ctx": em2, "f1_with_ctx": f12,
                "em_with_1mem": em3, "f1_with_1mem": f13,
                "context_len_tokens": ctx_len_tokens_full,
                "mem_recon_acc": recon_best,
                "mem_trunc_info": info,
            })

        # 清理显存碎片
        del compressor
        torch.cuda.empty_cache()

        if args.verbose and (i+1) % 10 == 0:
            print(f"[{i+1}/{args.n}] EM/F1(no-ctx)={em1:.0f}/{f11:.1f}  EM/F1(+ctx)={em2:.0f}/{f12:.1f}  EM/F1(+1mem)={em3:.0f}/{f13:.1f}  recon={recon_best*100:.1f}%")

    N = max(1, len(rows))
    print("\n=== Final ===")
    print(f"(1) No context     : EM={em_1/N*100:.1f}  F1={f1_1/N*100:.1f}")
    print(f"(2) With gold ctx  : EM={em_2/N*100:.1f}  F1={f1_2/N*100:.1f}")
    print(f"(3) With 1 mem     : EM={mem_em/N*100:.1f}  F1={mem_f1/N*100:.1f}  (mem QA，仅统计)")
    print(f"Saving CSV -> {args.csv_out}")

    with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    if jsonl is not None:
        with open(args.save_preds, "w", encoding="utf-8") as f:
            for r in jsonl:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL -> {args.save_preds}")

if __name__ == "__main__":
    main()
