#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
三种评测模式（均用原始 NQ 的 context + gold QA）：
1) no_ctx     : 无上下文 QA（最差基线）
2) gold_ctx   : 原始 context + question（理想上限）
3) mem_token  : 先用原始 context 训练 mem（复读），再用“变种 QA”（来自 aug.jsonl，过滤与原题问句相同的问句，
                且答案需与 gold 有足够重叠）对 mem 做 QA 对齐（仅更新 mem），推理时 [BOS][mem][prompt]

关键点：
- 三阶段同构：复读训练 / QA 对齐（student）/ 推理，全用 [BOS] + [mem] + emb(...)
- 避免双 BOS：tokenizer 编码后若首位是 BOS，先剥掉，再手动补 BOS 的 embedding
- 学生侧强制对齐：确保 inputs_embeds 与 labels 序列长度一致，避免 off-by-one 报错
- 每次运行创建新目录，日志带时间戳；导出 CSV/JSONL/XLSX
"""

import argparse, csv, json, re, string, random, math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd

# ===================== Normalize & Metrics =====================
_ARTICLES = {"a","an","the"}
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

# ===================== Overlap helpers =====================
def jaccard_overlap(a: str, b: str) -> float:
    A = set(normalize_answer(a).split()); B = set(normalize_answer(b).split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def has_answer_overlap(ans: str, golds: List[str], thr: float) -> bool:
    return any(jaccard_overlap(ans, g) >= thr for g in golds)

# ===================== Prompts & Generation =====================
SYS_PROMPT = "You are a helpful, concise QA assistant."

def build_prompt(tokenizer, question: str, context: Optional[str] = None) -> str:
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
        messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System:\n{SYS_PROMPT}\n\n### User:\n{user}\n\n### Assistant:"

def build_prompt_for_mem(tokenizer, question: str) -> str:
    user = (
        "You have been prepended with special MEMORY tokens that compress relevant context.\n"
        "Answer the question concisely using only that MEMORY. If it's not answerable, output exactly: I don't know.\n\n"
        f"Question: {question}\nAnswer:"
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System:\n{SYS_PROMPT}\n\n### User:\n{user}\n\n### Assistant:"

def postprocess_model_answer(text: str) -> str:
    if not text: return ""
    text = text.strip()
    if not text: return ""
    ans = text.splitlines()[0].strip()
    ans = re.sub(r"^(answer\s*:)\s*", "", ans, flags=re.I)
    ans = re.sub(r"[\"“”‘’]+$", "", ans).strip()
    if len(ans.split()) > 12:
        ans = " ".join(ans.split()[:12])
    return ans

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
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return postprocess_model_answer(text)

# for gold_ctx span alignment
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
    best, best_score = pred, -1.0
    for L in range(1, min(max_len, len(tokens))+1):
        for i in range(0, len(tokens)-L+1):
            span = " ".join(tokens[i:i+L])
            score = _f1_tokens(pred_tokens, normalize_answer(span).split())
            if score > best_score:
                best_score, best = score, span
    return best

def strip_leading_bos(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    if tokenizer.bos_token_id is not None and input_ids.shape[1] >= 1 and int(input_ids[0,0]) == tokenizer.bos_token_id:
        return input_ids[:,1:]
    return input_ids

# ===================== Logger =====================
class SimpleLogger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")
    def __call__(self, s: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {s}"
        print(line); self.f.write(line+"\n"); self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass

# ===================== Mem Compressor =====================
class MemCompressor:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, mem_len: int=1, enable_gc: bool=False, logger=None):
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
        print(s) if self.logger is None else self.logger(s)

    def _enter(self):
        self._saved_train_flag = self.model.training
        self._saved_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False
        if self.enable_gc and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.train()

    def _exit(self):
        self.model.config.use_cache = self._saved_use_cache
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if self._saved_train_flag is False: self.model.eval()

    def _build_inputs(self, ctx_ids: torch.LongTensor):
        """复读训练：inputs_embeds = [BOS] + [mem] + emb(ctx)；labels 对 [BOS][mem] 置 -100。"""
        ctx_ids = ctx_ids.to(self.model.device)
        ctx_ids_b = ctx_ids.unsqueeze(0)          # [1, T]
        tok = self.emb(ctx_ids_b)                  # [1, T, H]
        mem_cast = self.mem.to(dtype=tok.dtype)
        if self.tokenizer.bos_token_id is not None:
            bos_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.model.device)
            bos_emb = self.emb(bos_ids)           # [1,1,H]
            inputs_embeds = torch.cat([bos_emb, mem_cast, tok], dim=1)
            mask_len = 1 + self.mem.shape[1]
        else:
            inputs_embeds = torch.cat([mem_cast, tok], dim=1)
            mask_len = self.mem.shape[1]
        labels = torch.cat([
            torch.full((1, mask_len), -100, dtype=torch.long, device=ctx_ids.device),
            ctx_ids_b
        ], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=ctx_ids.device)
        return inputs_embeds, labels, attn

    def train_mem(self, ctx_ids: torch.LongTensor, steps=1000, lr=1e-2, weight_decay=1e-2,
                  beta1=0.9, beta2=0.9, target_acc=1.0, log_every=100, verbose=False):
        for p in self.model.parameters(): p.requires_grad_(False)
        opt = AdamW([{"params":[self.mem], "lr":lr, "weight_decay":weight_decay, "betas":(beta1,beta2)}])
        best_acc, best_mem = 0.0, None
        self._enter()
        try:
            for step in range(1, steps+1):
                opt.zero_grad(set_to_none=True)
                inputs_embeds, labels, attn = self._build_inputs(ctx_ids)
                # 防呆对齐
                labels = _ensure_same_len(inputs_embeds, labels)
                out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
                loss = out.loss
                loss.backward(); opt.step()
                with torch.no_grad(): acc = self.recon_acc(ctx_ids)
                if acc > best_acc: best_acc, best_mem = acc, self.mem.detach().clone()
                if verbose and (step % log_every == 0 or step == 1):
                    self._log(f"[mem-train] step {step}/{steps} loss={loss.item():.4f} recon_acc={acc*100:.2f}%")
                if acc >= target_acc:
                    if verbose: self._log(f"[mem-train] early stop at {step} recon_acc={acc*100:.2f}%")
                    break
        finally:
            self._exit()
        if best_mem is not None:
            with torch.no_grad(): self.mem.copy_(best_mem)
        return best_acc

    @torch.no_grad()
    def recon_acc(self, ctx_ids: torch.LongTensor) -> float:
        self.model.eval()
        inputs_embeds, labels, attn = self._build_inputs(ctx_ids)
        labels = _ensure_same_len(inputs_embeds, labels)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn)
        logits = out.logits
        pred = logits[:, :-1, :].argmax(dim=-1)
        tgt = labels[:, 1:]
        mask = (tgt != -100)
        correct = (pred[mask] == tgt[mask]).float().sum().item()
        total = int(mask.sum().item())
        return (correct / total) if total > 0 else 0.0

    def qa_align_mem(self, qas: List[Dict[str,str]], orig_question: str, teacher_context: str,
                     tokenizer, max_steps=200, lr=1e-2, T=2.0, lambda_ce=1.0, lambda_kd=1.0,
                     log_every=50, verbose=False,
                     patience: int = 80, min_improve: float = 1e-3,
                     clip_norm: float = 1.0, anchor_l2: float = 5e-4,
                     use_cosine: bool = True):
        """只更新 mem；student 与推理同构：[BOS][mem][prompt+answer]；teacher=原文+问。"""
        q0_norm = normalize_answer(orig_question)
        pool = [qa for qa in qas if qa.get("question") and qa.get("answer") and normalize_answer(qa["question"]) != q0_norm]
        if not pool: return 0.0, 0

        for p in self.model.parameters(): p.requires_grad_(False)
        opt = AdamW([{"params":[self.mem], "lr":lr, "weight_decay":0.0, "betas":(0.9,0.9)}])

        mem_init = self.mem.detach().clone()
        best_mem = self.mem.detach().clone()
        best_metric = float("inf")
        bad_rounds = 0
        ema = None

        def build_teacher(q: str, a: str):
            p = build_prompt(tokenizer, q, context=teacher_context)
            ids = tokenizer(p + a, return_tensors="pt")
            ans_ids = tokenizer(a, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            labels = ids["input_ids"].clone()
            labels[:, : labels.shape[1] - ans_ids.shape[0]] = -100
            return {k: v.to(self.model.device) for k, v in ids.items()}, labels.to(self.model.device)

        def build_student(q: str, a: str):
            """
            student 输入与推理完全同构：
              inputs_embeds = [BOS] + [mem] + emb(input_ids_rest)
              labels        =   -100   -100..  + labels_rest
            其中 input_ids_rest 与 labels_rest 都是对 tokenizer 输出统一“剥掉首 BOS”后的同一长度视图。
            """
            p = build_prompt_for_mem(tokenizer, q)
            ids_full = tokenizer(p + a, return_tensors="pt")["input_ids"].to(self.model.device)  # [1, L_full]
            has_bos = (tokenizer.bos_token_id is not None
                       and ids_full.shape[1] >= 1
                       and int(ids_full[0, 0]) == tokenizer.bos_token_id)
            input_ids_rest = ids_full[:, 1:] if has_bos else ids_full                     # [1, L_rest]

            ans_ids = tokenizer(a, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
            labels_full = ids_full.clone()
            labels_full[:, : labels_full.shape[1] - ans_ids.shape[0]] = -100
            labels_rest = labels_full[:, 1:] if has_bos else labels_full                  # [1, L_rest]
            return input_ids_rest, labels_rest, has_bos

        for step in range(1, max_steps+1):
            qa = pool[(step-1) % len(pool)]
            q, a = qa["question"], qa["answer"]
            if not a: continue

            # Cosine decay
            if use_cosine and max_steps > 1:
                lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * (step-1) / (max_steps-1)))
                for g in opt.param_groups: g["lr"] = lr_t
            else:
                lr_t = lr

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                t_in, t_lab = build_teacher(q, a)
                out_T = self.model(**t_in, labels=t_lab)
                logits_T = out_T.logits[t_lab != -100].view(-1, out_T.logits.size(-1))

            ids_rest, s_lab_rest, has_bos = build_student(q, a)
            tok_emb = self.emb(ids_rest)                        # [1, L_rest, H]
            mem_cast = self.mem.to(dtype=tok_emb.dtype)

            if has_bos:
                bos_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.model.device)
                bos_emb = self.emb(bos_ids)                    # [1,1,H]
                inputs_embeds = torch.cat([bos_emb, mem_cast, tok_emb], dim=1)
                mask_len = 1 + self.mem.shape[1]
            else:
                inputs_embeds = torch.cat([mem_cast, tok_emb], dim=1)
                mask_len = self.mem.shape[1]

            labels = torch.cat([
                torch.full((1, mask_len), -100, dtype=torch.long, device=self.model.device),
                s_lab_rest
            ], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.model.device)

            # 防呆：长度强制一致
            labels = _ensure_same_len(inputs_embeds, labels)

            out_S = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
            logits_S = out_S.logits[labels != -100].view(-1, out_S.logits.size(-1))

            # CE
            ce = torch.nn.functional.cross_entropy(logits_S, labels[labels != -100])

            # KD（若答案分词长度偶发不一致，做 min_len 对齐）
            if logits_S.size(0) != logits_T.size(0):
                L = min(logits_S.size(0), logits_T.size(0))
                logits_S_kd = logits_S[:L]
                logits_T_kd = logits_T[:L]
            else:
                logits_S_kd, logits_T_kd = logits_S, logits_T
            log_pS = torch.nn.functional.log_softmax(logits_S_kd / T, dim=-1)
            pT = torch.nn.functional.softmax(logits_T_kd / T, dim=-1)
            kd = torch.nn.functional.kl_div(log_pS, pT, reduction="batchmean") * (T*T)

            # 稳定项：锚定 mem 变化
            l2 = torch.tensor(0.0, device=self.model.device)
            if anchor_l2 > 0:
                l2 = torch.mean((self.mem - mem_init).pow(2)) * anchor_l2

            loss = lambda_ce*ce + lambda_kd*kd + l2
            loss.backward()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([self.mem], max_norm=clip_norm)
            opt.step()

            # EMA + early stop
            loss_val = float(loss.item())
            ema = loss_val if ema is None else (0.9*ema + 0.1*loss_val)
            improved = (best_metric - ema) > min_improve
            if improved:
                best_metric = ema
                best_mem = self.mem.detach().clone()
                bad_rounds = 0
            else:
                bad_rounds += 1

            if verbose and (step % log_every == 0 or step == 1):
                self._log(f"[qa-align] step {step}/{max_steps} loss={loss_val:.4f} ema={ema:.4f} (ce={ce.item():.4f}, kd={kd.item():.4f}, l2={l2.item():.4f}) lr={lr_t:.2e}")

            if patience and bad_rounds >= patience:
                if verbose: self._log(f"[qa-align] early stop at step {step} (no improve for {patience} steps).")
                break

        with torch.no_grad():
            self.mem.copy_(best_mem)

        return (best_metric if best_metric != float('inf') else (ema or 0.0)), step

    @torch.inference_mode()
    def answer_with_mem(self, tokenizer: AutoTokenizer, question: str, max_new_tokens=16, temperature=0.0) -> str:
        self.model.eval()
        prompt = build_prompt_for_mem(tokenizer, question)
        toks = tokenizer(prompt, return_tensors="pt")
        input_ids = toks["input_ids"].to(self.model.device)
        rest_ids = strip_leading_bos(tokenizer, input_ids)
        rest_emb = self.emb(rest_ids)
        mem_cast = self.mem.to(dtype=rest_emb.dtype)
        if tokenizer.bos_token_id is not None:
            bos_ids = torch.tensor([[tokenizer.bos_token_id]], device=self.model.device)
            bos_emb = self.emb(bos_ids)
            inputs_embeds = torch.cat([bos_emb, mem_cast, rest_emb], dim=1)
        else:
            inputs_embeds = torch.cat([mem_cast, rest_emb], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.model.device)
        gen_ids = self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False if temperature<=0 else True,
            temperature=temperature if temperature>0 else None, top_p=1.0, num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
        )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return postprocess_model_answer(text)

# ===================== Safety: ensure same length =====================
def _ensure_same_len(inputs_embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    L_in = inputs_embeds.size(1)
    L_lb = labels.size(1)
    if L_lb > L_in:
        labels = labels[:, :L_in]
    elif L_lb < L_in:
        pad = torch.full((labels.size(0), L_in - L_lb), -100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([labels, pad], dim=1)
    return labels

# ===================== Helpers =====================
def _extract_keywords(q: str) -> List[str]:
    kws = re.findall(r"[A-Za-z0-9]{3,}", q.lower()); seen=set(); out=[]
    for w in kws:
        if w not in seen: seen.add(w); out.append(w)
    return out

def truncate_ctx_for_mem_training(question: str, context: str, tokenizer, max_ctx_tokens: int, mode="keyword", verbose=False):
    full_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if full_ids.shape[0] <= max_ctx_tokens:
        return full_ids, None
    if mode == "head":
        s = 0
    else:
        ctx_lc, pos = context.lower(), None
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

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()

    # 模型 & 推理
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","auto"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--enable_gc", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")

    # NQ 数据（评测 gold 来自这里）
    ap.add_argument("--nq_split", type=str, default="test")

    # 增强数据（仅用于 QA 对齐）
    ap.add_argument("--aug_path", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qa_limit_per_doc", type=int, default=4)
    ap.add_argument("--qa_answer_overlap", type=float, default=0.6, help="增强 QA 答案与 gold 的 Jaccard 重叠阈值(0~1)")

    # mem 训练 & QA 对齐
    ap.add_argument("--mem_len", type=int, default=1)
    ap.add_argument("--mem_steps", type=int, default=1000)
    ap.add_argument("--mem_lr", type=float, default=1e-2)
    ap.add_argument("--mem_wd", type=float, default=1e-2)
    ap.add_argument("--mem_beta1", type=float, default=0.9)
    ap.add_argument("--mem_beta2", type=float, default=0.9)
    ap.add_argument("--mem_target_acc", type=float, default=1.0)
    ap.add_argument("--mem_log_every", type=int, default=100)
    ap.add_argument("--max_ctx_tokens", type=int, default=1024)
    ap.add_argument("--ctx_window", type=str, default="keyword", choices=["keyword","head"])

    ap.add_argument("--qa_align", action="store_true")
    ap.add_argument("--qa_steps", type=int, default=200)
    ap.add_argument("--qa_lr", type=float, default=1e-2)
    ap.add_argument("--kd_T", type=float, default=4.0)
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--lambda_kd", type=float, default=0.25)

    # 稳态超参
    ap.add_argument("--qa_patience", type=int, default=80, help="QA 对齐早停等待步数")
    ap.add_argument("--qa_min_improve", type=float, default=1e-3, help="EMA loss 至少改善阈值")
    ap.add_argument("--qa_clip_norm", type=float, default=1.0, help="梯度裁剪阈值")
    ap.add_argument("--qa_anchor_l2", type=float, default=5e-4, help="L2 锚定到 mem 初值的权重")
    ap.add_argument("--qa_cosine", action="store_true", help="启用余弦退火学习率")

    # 输出
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--csv_name", type=str, default="preds.csv")
    ap.add_argument("--jsonl_name", type=str, default="preds.jsonl")
    ap.add_argument("--xlsx_name", type=str, default="preds.xlsx")

    # 模式
    ap.add_argument("--modes", type=str, default="no_ctx,gold_ctx,mem_token",
                    help="逗号分隔: 可选 no_ctx,gold_ctx,mem_token")

    args = ap.parse_args()

    # ===== 独立 run 目录 =====
    model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", Path(args.model).name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nflag = f"n{args.limit}" if args.limit is not None else "nall"
    run_dir = Path(args.out_dir) / f"run_{ts}_{model_slug}_{nflag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / args.csv_name
    jsonl_path = run_dir / args.jsonl_name
    xlsx_path = run_dir / args.xlsx_name
    log_path = run_dir / "train.log"
    cfg_path = run_dir / "config.json"

    logger = SimpleLogger(log_path)
    with open(cfg_path, "w", encoding="utf-8") as f: json.dump(vars(args), f, ensure_ascii=False, indent=2)
    logger(f"[run-dir] {run_dir}")

    # 随机种子
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 加载 NQ（评测 gold）
    ds = load_dataset("LLukas22/nq-simplified", split=args.nq_split)
    nq_map: Dict[int, Dict[str, Any]] = {}
    for idx, ex in enumerate(ds):
        nq_map[idx] = {"question": ex["question"], "context": ex["context"], "answers": ex.get("answers", {}).get("text", [])}

    # 读取增强数据
    aug_recs: List[Dict[str, Any]] = []
    with open(args.aug_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            aug_recs.append(json.loads(line))
    if args.limit is not None: aug_recs = aug_recs[: args.limit]
    logger(f"[data] aug={len(aug_recs)}  nq_split='{args.nq_split}' size={len(ds)}")

    # 解析模式
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        assert m in {"no_ctx","gold_ctx","mem_token"}, f"非法模式: {m}"

    # 模型
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda" if args.device == "cuda" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 统计器
    agg = {m: {"em":0.0,"f1":0.0,"n":0} for m in ["no_ctx","gold_ctx","mem_token"]}

    header = [
        "id","question","gold_answers","context_len_tokens",
        "pred_no_ctx","em_no_ctx","f1_no_ctx",
        "pred_gold_ctx","em_gold_ctx","f1_gold_ctx",
        "pred_mem_token","em_mem_token","f1_mem_token",
        "mem_recon_acc","qa_align_loss_best","qa_align_steps","qa_used"
    ]
    rows, jrows = [], []

    for i, rec in enumerate(aug_recs):
        ex_id = int(rec.get("id", -1))
        base = nq_map.get(ex_id)
        if base is None:
            logger(f"[warn] skip id={ex_id} (not in NQ split)")
            continue

        q_orig = base["question"]
        ctx_orig = base["context"]
        golds = base["answers"] or []
        gold_str = " || ".join(golds)

        # token 长度
        ctx_ids_full = tokenizer(ctx_orig, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        ctx_len_tokens = int(ctx_ids_full.shape[0])

        # (1) no_ctx
        pred_no_ctx = em_no_ctx = f1_no_ctx = ""
        if "no_ctx" in modes:
            prompt1 = build_prompt(tokenizer, q_orig, context=None)
            p1 = generate_answer(model, tokenizer, prompt1, args.max_new_tokens, args.temperature)
            if golds:
                em1, f11 = metric_em_f1(p1, golds)
                agg["no_ctx"]["em"] += em1; agg["no_ctx"]["f1"] += f11; agg["no_ctx"]["n"] += 1
                em_no_ctx, f1_no_ctx = f"{em1:.3f}", f"{f11:.3f}"
            pred_no_ctx = p1

        # (2) gold_ctx
        pred_gold_ctx = em_gold_ctx = f1_gold_ctx = ""
        if "gold_ctx" in modes:
            prompt2 = build_prompt(tokenizer, q_orig, context=ctx_orig)
            p2_raw = generate_answer(model, tokenizer, prompt2, args.max_new_tokens, args.temperature)
            p2 = spanize_to_context(p2_raw, ctx_orig)
            if golds:
                em2, f12 = metric_em_f1(p2, golds)
                agg["gold_ctx"]["em"] += em2; agg["gold_ctx"]["f1"] += f12; agg["gold_ctx"]["n"] += 1
                em_gold_ctx, f1_gold_ctx = f"{em2:.3f}", f"{f12:.3f}"
            pred_gold_ctx = p2

        # (3) mem_token
        pred_mem = em_mem = f1_mem = ""
        recon_best = 0.0; qa_loss_best = 0.0; qa_steps_done = 0; qa_used = 0
        if "mem_token" in modes:
            # 复读训练只看截断窗口
            ctx_ids_for_mem, _ = truncate_ctx_for_mem_training(
                q_orig, ctx_orig, tokenizer, args.max_ctx_tokens, args.ctx_window, verbose=False
            )
            ctx_ids_for_mem = ctx_ids_for_mem.to(model.device)

            compressor = MemCompressor(model, tokenizer, mem_len=args.mem_len, enable_gc=args.enable_gc, logger=logger if args.verbose else None)
            recon_best = compressor.train_mem(
                ctx_ids=ctx_ids_for_mem, steps=args.mem_steps, lr=args.mem_lr,
                weight_decay=args.mem_wd, beta1=args.mem_beta1, beta2=args.mem_beta2,
                target_acc=args.mem_target_acc, log_every=args.mem_log_every, verbose=args.verbose,
            )

            # 变种 QA：过滤原题问句+答案重叠
            qas_all = rec.get("qas", []) or []
            q0_norm = normalize_answer(q_orig)
            qas_pool = []
            for qa in qas_all:
                qq, aa = qa.get("question"), qa.get("answer")
                if not qq or not aa: continue
                if normalize_answer(qq) == q0_norm: continue
                if args.qa_answer_overlap > 0.0 and not has_answer_overlap(aa, golds, args.qa_answer_overlap):
                    continue
                qas_pool.append(qa)
            if args.qa_limit_per_doc is not None and len(qas_pool) > args.qa_limit_per_doc:
                qas_pool = qas_pool[: args.qa_limit_per_doc]

            if args.qa_align and qas_pool:
                qa_used = len(qas_pool)
                qa_loss_best, qa_steps_done = compressor.qa_align_mem(
                    qas=qas_pool, orig_question=q_orig, teacher_context=ctx_orig, tokenizer=tokenizer,
                    max_steps=args.qa_steps, lr=args.qa_lr, T=args.kd_T,
                    lambda_ce=args.lambda_ce, lambda_kd=args.lambda_kd,
                    log_every=max(1, args.qa_steps//4), verbose=args.verbose,
                    patience=args.qa_patience, min_improve=args.qa_min_improve,
                    clip_norm=args.qa_clip_norm, anchor_l2=args.qa_anchor_l2,
                    use_cosine=args.qa_cosine
                )

            # 推理（与 student 同构）
            p3 = compressor.answer_with_mem(tokenizer, q_orig, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            if golds:
                em3, f13 = metric_em_f1(p3, golds)
                agg["mem_token"]["em"] += em3; agg["mem_token"]["f1"] += f13; agg["mem_token"]["n"] += 1
                em_mem, f1_mem = f"{em3:.3f}", f"{f13:.3f}"
            pred_mem = p3

            del compressor
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        rows.append([
            ex_id, q_orig, gold_str, ctx_len_tokens,
            pred_no_ctx, em_no_ctx, f1_no_ctx,
            pred_gold_ctx, em_gold_ctx, f1_gold_ctx,
            pred_mem, em_mem, f1_mem,
            f"{recon_best:.3f}", f"{qa_loss_best:.4f}", qa_steps_done, qa_used
        ])
        jrows.append({
            "id": ex_id, "question": q_orig, "gold_answers": golds,
            "pred_no_ctx": pred_no_ctx, "em_no_ctx": em_no_ctx, "f1_no_ctx": f1_no_ctx,
            "pred_gold_ctx": pred_gold_ctx, "em_gold_ctx": em_gold_ctx, "f1_gold_ctx": f1_gold_ctx,
            "pred_mem_token": pred_mem, "em_mem_token": em_mem, "f1_mem_token": f1_mem,
            "mem_recon_acc": recon_best, "qa_align_loss_best": qa_loss_best, "qa_align_steps": qa_steps_done,
            "context_len_tokens": ctx_len_tokens, "qa_used": qa_used,
        })

        if args.verbose and (i+1) % 20 == 0:
            logger(f"[{i+1}/{len(aug_recs)}] processed.")

    # 汇总
    logger("\n=== Final Averages ===")
    summary_rows = []
    for m in ["no_ctx","gold_ctx","mem_token"]:
        if m not in modes: continue
        n = max(1, agg[m]["n"])
        em = agg[m]["em"]/n if agg[m]["n"]>0 else 0.0
        f1 = agg[m]["f1"]/n if agg[m]["n"]>0 else 0.0
        logger(f"{m:>9} : EM={em*100:.1f}  F1={f1*100:.1f}  (N={agg[m]['n']})")
        summary_rows.append({"mode": m, "EM": em, "F1": f1, "N": agg[m]["n"]})

    # 写盘
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow([
            "id","question","gold_answers","context_len_tokens",
            "pred_no_ctx","em_no_ctx","f1_no_ctx",
            "pred_gold_ctx","em_gold_ctx","f1_gold_ctx",
            "pred_mem_token","em_mem_token","f1_mem_token",
            "mem_recon_acc","qa_align_loss_best","qa_align_steps","qa_used"
        ]); writer.writerows(rows)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in jrows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        import openpyxl  # noqa: F401
        df = pd.DataFrame(rows, columns=[
            "id","question","gold_answers","context_len_tokens",
            "pred_no_ctx","em_no_ctx","f1_no_ctx",
            "pred_gold_ctx","em_gold_ctx","f1_gold_ctx",
            "pred_mem_token","em_mem_token","f1_mem_token",
            "mem_recon_acc","qa_align_loss_best","qa_align_steps","qa_used"
        ])
        df_sum = pd.DataFrame(summary_rows)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_sum.to_excel(writer, index=False, sheet_name="summary")
            df.to_excel(writer, index=False, sheet_name="preds")
    except Exception as e:
        logger(f"[warn] write xlsx failed: {e}")

    logger(f"Saved CSV -> {csv_path}")
    logger(f"Saved JSONL -> {jsonl_path}")
    logger(f"Saved XLSX -> {xlsx_path}")
    logger.close()

if __name__ == "__main__":
    main()
