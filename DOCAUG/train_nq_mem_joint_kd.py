#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NQ 三种模式训练/评测：
(1) 无上下文；(2) 有金上下文；(3) Mem(联合训练：AE重建+QA问答)，评测仅用原始问题。

更新：
- 新增 “答案 token 的 Logit 蒸馏 (KD)” 到 QA 分支，teacher 读取 gold context，student 读取 mem。
- 通过命令行参数控制 KD 温度、权重、前 K 个答案 token 的额外权重、是否启用 KD；
- mem token 个数仍由 --mem_len 控制。
- 其余流程保持不变（尽量最小侵入）。

数据：
  --raw_subset_file: data/nq_subset/raw/nq_subset_100.jsonl   # 含原问题
  --aug_file       : data/nq_subset/augmented/nq_multiqa_only_no_unans.jsonl  # 只含新问题
"""

import argparse, csv, json, re, string, random, os
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
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

def build_memqa_prompt(tokenizer, question: str) -> str:
    user = (
        "Answer the question with the shortest exact phrase you can recall. "
        "If you don't know, output exactly: I don't know.\n\n"
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
# 上下文截断（仅用于 mem 训练）
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
    verbose: bool = False
):
    full_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if full_ids.shape[0] <= max_ctx_tokens:
        return full_ids, None

    if mode == "head":
        s = 0
    else:
        ctx_lc = context.lower()
        pos = None
        for kw in _extract_keywords(question):
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
    info = f"[mem-trunc] full={full_ids.shape[0]} keep={max_ctx_tokens} range=({s},{e}) mode={mode}"
    if verbose:
        print(info)
    return ids_trunc.to("cpu"), info

# =========================
# ③ Mem：联合训练 AE + QA (+ 可选 KD)
# =========================

class MemCompressor:
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
        device = self.model.device
        ctx_ids = ctx_ids.to(device)
        ctx_ids_b = ctx_ids.unsqueeze(0)
        tok_embeds = self.emb(ctx_ids_b)
        mem_cast = self.mem.to(dtype=tok_embeds.dtype)
        inputs_embeds = torch.cat([mem_cast, tok_embeds], dim=1)
        labels = torch.cat(
            [torch.full((1, self.mem.shape[1]), -100, dtype=torch.long, device=device), ctx_ids_b],
            dim=1
        )
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        return inputs_embeds, labels, attn

    def _build_mem_qa_inputs(self, prompt_ids: torch.LongTensor, answer_ids: torch.LongTensor):
        device = self.model.device
        prompt_ids = prompt_ids.to(device).unsqueeze(0)
        answer_ids = answer_ids.to(device).unsqueeze(0)
        emb_p = self.emb(prompt_ids)
        emb_a = self.emb(answer_ids)
        mem_cast = self.mem.to(dtype=emb_p.dtype)
        inputs_embeds = torch.cat([mem_cast, emb_p, emb_a], dim=1)
        labels = torch.cat(
            [torch.full((1, self.mem.shape[1] + prompt_ids.shape[1]), -100, dtype=torch.long, device=device), answer_ids],
            dim=1
        )
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        return inputs_embeds, labels, attn

    def _sample_qa_pair(self, qa_pairs: List[Dict], step: int):
        if not qa_pairs: return None
        idx = step % len(qa_pairs)
        return qa_pairs[idx]

    def train_mem(self,
                  ctx_ids: torch.LongTensor,
                  qa_pairs: Optional[List[Dict]] = None,
                  steps: int = 500,
                  lr: float = 1e-2,
                  weight_decay: float = 1e-2,
                  beta1: float = 0.9,
                  beta2: float = 0.9,
                  alpha_start: float = 0.8,
                  alpha_end: float = 0.4,
                  mem_l2: float = 1e-4,
                  noise_std: float = 0.0,
                  log_every: int = 100,
                  verbose: bool = False,
                  # KD 相关
                  enable_kd: bool = False,
                  context_text: Optional[str] = None,
                  teacher_model: Optional[AutoModelForCausalLM] = None,
                  teacher_tokenizer: Optional[AutoTokenizer] = None,
                  kd_temp: float = 2.0,
                  kd_lambda: float = 1.0,
                  kd_first_k: int = 0,
                  kd_head_weight: float = 1.0,
                  ):
        for p in self.model.parameters(): p.requires_grad_(False)
        opt = AdamW([{"params": [self.mem], "lr": lr, "weight_decay": weight_decay, "betas": (beta1, beta2)}])

        kl = torch.nn.KLDivLoss(reduction="batchmean")

        best_acc = 0.0
        best_mem = None
        self._enter_mem_train_mode()
        try:
            for step in range(1, steps + 1):
                t = (step - 1) / max(1, steps - 1)
                alpha = alpha_start + (alpha_end - alpha_start) * t

                opt.zero_grad(set_to_none=True)

                # AE
                inputs_embeds, labels, attn = self._build_mem_training_inputs(ctx_ids)
                if noise_std > 0:
                    inputs_embeds[:, :self.mem.shape[1], :] += torch.randn_like(inputs_embeds[:, :self.mem.shape[1], :]) * noise_std
                out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
                loss_ae = out.loss

                # QA
                loss_qa = torch.tensor(0.0, device=self.model.device)
                loss_kd = torch.tensor(0.0, device=self.model.device)
                if qa_pairs:
                    pair = self._sample_qa_pair(qa_pairs, step)
                    if pair is not None:
                        pr_ids, ans_ids = pair["pr_ids"], pair["ans_ids"]
                        qa_inp, qa_lab, qa_attn = self._build_mem_qa_inputs(pr_ids, ans_ids)
                        if noise_std > 0:
                            qa_inp[:, :self.mem.shape[1], :] += torch.randn_like(qa_inp[:, :self.mem.shape[1], :]) * noise_std
                        out2 = self.model(inputs_embeds=qa_inp, attention_mask=qa_attn, labels=qa_lab)
                        loss_qa = out2.loss

                        # ===== KD: 答案 token 的 logit 蒸馏 =====
                        if enable_kd and (teacher_model is not None) and (teacher_tokenizer is not None) and (context_text is not None) and kd_lambda > 0:
                            q_text = pair.get("q_text", None)
                            a_text = pair.get("a_text", None)
                            if q_text and a_text:
                                try:
                                    # 学生答案对齐：取预测答案 token 的对应 logits 切片
                                    answer_len = ans_ids.shape[0]
                                    start_pos = self.mem.shape[1] + pr_ids.shape[0] - 1
                                    logits_s = out2.logits[0, start_pos : start_pos + answer_len, :]  # [Ta, V]

                                    # 老师构造：gold ctx + prompt（span 复制风格） + 答案文本（teacher forcing）
                                    t_prompt = build_prompt(teacher_tokenizer, q_text, context=context_text)
                                    t_pr_ids = teacher_tokenizer(t_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
                                    t_ans_ids = teacher_tokenizer(a_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.model.device)
                                    t_input_ids = torch.cat([t_pr_ids, t_ans_ids], dim=0).unsqueeze(0)
                                    with torch.no_grad():
                                        out_t = teacher_model(input_ids=t_input_ids)
                                    logits_t_full = out_t.logits[0]
                                    Tp, Ta = t_pr_ids.shape[0], t_ans_ids.shape[0]
                                    logits_t = logits_t_full[Tp - 1 : Tp - 1 + Ta, :]  # [Ta, V]

                                    # 温度 KD
                                    T = kd_temp if kd_temp > 0 else 1.0
                                    p_t = torch.softmax(logits_t / T, dim=-1)
                                    log_p_s = torch.log_softmax(logits_s / T, dim=-1)
                                    base_kd = (T * T) * kl(log_p_s, p_t)

                                    # 首 K 个答案 token 加权
                                    if kd_first_k > 0 and kd_head_weight != 1.0:
                                        k = min(kd_first_k, logits_s.shape[0])
                                        head_kd = (T * T) * kl(log_p_s[:k], p_t[:k])
                                        loss_kd = base_kd + (kd_head_weight - 1.0) * head_kd
                                    else:
                                        loss_kd = base_kd
                                except Exception as e:
                                    # 防御性：KD 失败不影响主流程
                                    if verbose:
                                        print(f"[warn] KD failed at step {step}: {e}")
                                    loss_kd = torch.tensor(0.0, device=self.model.device)

                # 总损失
                loss = alpha * loss_ae + (1 - alpha) * loss_qa + kd_lambda * loss_kd + mem_l2 * (self.mem.pow(2).mean())
                loss.backward()
                opt.step()

                with torch.no_grad():
                    acc = self.recon_acc(ctx_ids)

                if acc > best_acc:
                    best_acc = acc
                    best_mem = self.mem.detach().clone()

                if verbose and (step % log_every == 0 or step == 1):
                    print(f"[mem-train] step {step}/{steps} alpha={alpha:.2f} loss={loss.item():.4f} AE={loss_ae.item():.4f} QA={loss_qa.item():.4f} KD={loss_kd.item():.4f} recon={acc*100:.2f}%")

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
        prompt_embeds = self.emb(input_ids)
        mem_cast = self.mem.to(dtype=prompt_embeds.dtype)
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
# IO & Pair building
# =========================

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def normalize_text_for_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

# 注意：保留 memqa prompt 构造（student 用）
# teacher 使用 build_prompt(tokenizer, q, context)

def build_training_pairs_for_doc(tokenizer, context: str, qas: List[dict], orig_q: str) -> List[Dict]:
    pairs: List[Dict] = []
    orig_q_norm = normalize_text_for_key(orig_q or "")
    for qa in qas or []:
        q = (qa.get("q") or "").strip()
        a = (qa.get("a") or "").strip()
        a_span = bool(qa.get("a_span", True))
        if not q or not a or not a_span:
            continue
        if normalize_text_for_key(q) == orig_q_norm:
            continue
        if a not in context:
            continue
        prompt = build_memqa_prompt(tokenizer, q)
        pr_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ans_ids = tokenizer(a, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        pairs.append({
            "pr_ids": pr_ids,
            "ans_ids": ans_ids,
            "q_text": q,
            "a_text": a,
        })
    return pairs

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","auto"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--raw_subset_file", type=str, default="data/nq_subset/raw/nq_subset_100.jsonl")
    ap.add_argument("--aug_file", type=str, required=True)

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--csv_out", type=str, default="runs/nq_joint_results.csv")
    ap.add_argument("--save_preds", type=str, default="runs/nq_joint_preds.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 表示全量）")

    # mem 超参
    ap.add_argument("--mem_len", type=int, default=1, help="mem token 个数（可调）")
    ap.add_argument("--mem_steps", type=int, default=500)
    ap.add_argument("--mem_lr", type=float, default=1e-2)
    ap.add_argument("--mem_wd", type=float, default=1e-2)
    ap.add_argument("--mem_beta1", type=float, default=0.9)
    ap.add_argument("--mem_beta2", type=float, default=0.9)
    ap.add_argument("--alpha_start", type=float, default=0.8)
    ap.add_argument("--alpha_end", type=float, default=0.4)
    ap.add_argument("--mem_l2", type=float, default=1e-4)
    ap.add_argument("--mem_noise", type=float, default=0.0)
    ap.add_argument("--mem_log_every", type=int, default=100)
    ap.add_argument("--enable_gc", action="store_true")

    ap.add_argument("--max_ctx_tokens", type=int, default=1024)
    ap.add_argument("--ctx_window", type=str, default="keyword", choices=["keyword","head"])

    # ===== 新增：KD 控制项 =====
    ap.add_argument("--enable_kd", action="store_true", help="启用答案 token 的 logit 蒸馏")
    ap.add_argument("--teacher_model", type=str, default=None, help="teacher 模型（默认与 student 相同）")
    ap.add_argument("--kd_temp", type=float, default=2.0, help="KD 温度 T")
    ap.add_argument("--kd_lambda", type=float, default=1.0, help="KD 损失权重")
    ap.add_argument("--kd_first_k", type=int, default=0, help="前 K 个答案 token 额外加权（0 表示不加权）")
    ap.add_argument("--kd_head_weight", type=float, default=1.0, help="首 K token 的额外权重系数（=1 表示无额外权重）")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_preds), exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda" if args.device == "cuda" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Teacher（可选）
    teacher_model = None
    teacher_tokenizer = None
    if args.enable_kd:
        t_model_name = args.teacher_model if args.teacher_model is not None else args.model
        teacher_tokenizer = AutoTokenizer.from_pretrained(t_model_name, trust_remote_code=True)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            t_model_name, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        if teacher_tokenizer.pad_token_id is None and teacher_tokenizer.eos_token_id is not None:
            teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    raw = load_jsonl(args.raw_subset_file)
    aug = load_jsonl(args.aug_file)

    ctx2aug: Dict[str, dict] = {}
    for ex in aug:
        ctx = ex.get("context_orig") or ""
        if not ctx: continue
        ctx2aug[normalize_text_for_key(ctx)] = ex

    docs = []
    for ex in raw:
        ctx = ex.get("context") or ""
        q_orig = ex.get("question") or ""
        golds = ex.get("answers", [])
        key = normalize_text_for_key(ctx)
        aug_ex = ctx2aug.get(key, None)
        qas = aug_ex.get("qas", []) if aug_ex else []
        docs.append({"context": ctx, "q_orig": q_orig, "golds": golds, "qas": qas})

    if args.limit and args.limit > 0:
        docs = docs[:args.limit]

    header = [
        "id","context_len_tokens","question_orig","gold_answers",
        "pred_no_ctx","em_no_ctx","f1_no_ctx",
        "pred_with_ctx","em_with_ctx","f1_with_ctx",
        "pred_mem","em_mem","f1_mem",
        "mem_recon_acc","mem_train_qa_used"
    ]

    rows = []
    jsonl_rows = []
    em1_sum = f11_sum = em2_sum = f12_sum = 0.0
    em3_sum = f13_sum = 0.0

    for i, d in enumerate(docs):
        ctx = d["context"]; q_orig = d["q_orig"]; golds = d["golds"]

        ctx_ids_full = tokenizer(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        ctx_len_tokens_full = int(ctx_ids_full.shape[0])

        prompt1 = build_prompt(tokenizer, q_orig, context=None)
        pred1 = generate_answer(model, tokenizer, prompt1, args.max_new_tokens, args.temperature)
        em1, f11 = metric_em_f1(pred1, golds)

        prompt2 = build_prompt(tokenizer, q_orig, context=ctx)
        pred2_raw = generate_answer(model, tokenizer, prompt2, args.max_new_tokens, args.temperature)
        pred2_span = spanize_to_context(pred2_raw, ctx)
        em2, f12 = metric_em_f1(pred2_span, golds)

        ctx_ids_trunc, info = truncate_ctx_for_mem_training(
            q_orig, ctx, tokenizer, max_ctx_tokens=args.max_ctx_tokens, mode=args.ctx_window, verbose=args.verbose
        )
        ctx_ids_trunc = ctx_ids_trunc.to(model.device)

        qa_pairs = build_training_pairs_for_doc(tokenizer, ctx, d["qas"], q_orig)

        compressor = MemCompressor(model, tokenizer, mem_len=args.mem_len, enable_gc=args.enable_gc)
        recon_best = compressor.train_mem(
            ctx_ids=ctx_ids_trunc,
            qa_pairs=qa_pairs,
            steps=args.mem_steps,
            lr=args.mem_lr,
            weight_decay=args.mem_wd,
            beta1=args.mem_beta1,
            beta2=args.mem_beta2,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
            mem_l2=args.mem_l2,
            noise_std=args.mem_noise,
            log_every=args.mem_log_every,
            verbose=args.verbose,
            enable_kd=args.enable_kd,
            context_text=ctx,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            kd_temp=args.kd_temp,
            kd_lambda=args.kd_lambda,
            kd_first_k=args.kd_first_k,
            kd_head_weight=args.kd_head_weight,
        )

        prompt_memqa = build_memqa_prompt(tokenizer, q_orig)
        pred_mem = compressor.answer_with_mem(prompt_memqa, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        em3, f13 = metric_em_f1(pred_mem, golds)

        rows.append([
            i, ctx_len_tokens_full, q_orig, " || ".join(golds),
            pred1, f"{em1:.3f}", f"{f11:.3f}",
            pred2_span, f"{em2:.3f}", f"{f12:.3f}",
            pred_mem, f"{em3:.3f}", f"{f13:.3f}",
            f"{recon_best:.3f}", len(qa_pairs)
        ])
        jsonl_rows.append({
            "id": i,
            "question_orig": q_orig,
            "context": ctx,
            "gold_answers": golds,
            "pred_no_ctx": pred1, "em_no_ctx": em1, "f1_no_ctx": f11,
            "pred_with_ctx": pred2_span, "em_with_ctx": em2, "f1_with_ctx": f12,
            "pred_with_mem": pred_mem, "em_with_mem": em3, "f1_with_mem": f13,
            "context_len_tokens": ctx_len_tokens_full,
            "mem_recon_acc": recon_best,
            "mem_train_qa_used": len(qa_pairs),
            "mem_trunc_info": info,
        })

        em1_sum += em1; f11_sum += f11
        em2_sum += em2; f12_sum += f12
        em3_sum += em3; f13_sum += f13

        del compressor
        torch.cuda.empty_cache()

        if args.verbose and (i+1) % 10 == 0:
            print(f"[{i+1}/{len(docs)}] EM/F1(no-ctx)={em1:.0f}/{f11:.1f}  EM/F1(+ctx)={em2:.0f}/{f12:.1f}  EM/F1(+mem)={em3:.0f}/{f13:.1f}  recon={recon_best*100:.1f}%  trainQA={len(qa_pairs)}")

    N = max(1, len(rows))
    print("\n=== Final ===")
    print(f"(1) No context     : EM={em1_sum/N*100:.1f}  F1={f11_sum/N*100:.1f}")
    print(f"(2) With gold ctx  : EM={em2_sum/N*100:.1f}  F1={f12_sum/N*100:.1f}")
    print(f"(3) With mem (AE+QA): EM={em3_sum/N*100:.1f}  F1={f13_sum/N*100:.1f}")
    print(f"Saving CSV -> {args.csv_out}")

    with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow([
            "id","context_len_tokens","question_orig","gold_answers",
            "pred_no_ctx","em_no_ctx","f1_no_ctx",
            "pred_with_ctx","em_with_ctx","f1_with_ctx",
            "pred_mem","em_mem","f1_mem",
            "mem_recon_acc","mem_train_qa_used"
        ]); writer.writerows(rows)

    with open(args.save_preds, "w", encoding="utf-8") as f:
        for r in jsonl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved JSONL -> {args.save_preds}")

if __name__ == "__main__":
    main()

"""
运行示例：
python -u train_nq_mem_joint_kd.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct \
  --raw_subset_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/raw/nq_subset_100.jsonl \
  --aug_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/augmented/nq_multiqa_only_no_unans_first100.jsonl \
  --device cuda \
  --max_new_tokens 16 \
  --temperature 0.0 \
  --mem_len 1 \
  --mem_steps 400 \
  --mem_lr 1e-2 \
  --mem_wd 1e-2 \
  --alpha_start 0.8 \
  --alpha_end 0.4 \
  --mem_l2 1e-4 \
  --mem_noise 0.0 \
  --max_ctx_tokens 1024 \
  --ctx_window keyword \
  --enable_gc \
  --enable_kd \
  --kd_temp 2.0 \
  --kd_lambda 1.0 \
  --kd_first_k 3 \
  --kd_head_weight 2.0 \
  --csv_out runs/nq_joint_results_kd400step.csv \
  --save_preds runs/nq_joint_preds_kd400step.jsonl \
  --verbose \
  --limit 10
  
python -u train_nq_mem_joint_kd.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct \
  --raw_subset_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/raw/nq_subset_100_dif.jsonl \
  --aug_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/augmented/nq_multiqa_only_no_unans.jsonl \
  --device cuda \
  --max_new_tokens 16 \
  --temperature 0.0 \
  --mem_len 4 \
  --mem_steps 400 \
  --mem_lr 1e-2 \
  --mem_wd 1e-2 \
  --alpha_start 0.8 \
  --alpha_end 0.4 \
  --mem_l2 1e-4 \
  --mem_noise 0.0 \
  --max_ctx_tokens 1024 \
  --ctx_window keyword \
  --enable_gc \
  --enable_kd \
  --kd_temp 2.0 \
  --kd_lambda 1.0 \
  --kd_first_k 3 \
  --kd_head_weight 2.0 \
  --csv_out runs/nq_joint_results_kd400step4mem_dif.csv \
  --save_preds runs/nq_joint_preds_kd400step4mem_dif.jsonl \
  --verbose \
  --limit 10
"""