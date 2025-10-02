# 1) 国内建议先切镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2) 安装依赖
pip install -U datasets transformers accelerate torch

# 3) 跑评测（示例：本地或HF上的 Llama 8B Instruct）
python eval_llama_nq.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --n 100 \
  --max_new_tokens 64 \
  --temperature 0.0 \
  --save_preds preds_nq_llama8b.jsonl \
  --verbose
1.
python draft.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct \
  --n 100 \
  --max_new_tokens 64 \
  --temperature 0.0 \
  --save_preds preds_nq_llama8b_inst.jsonl \
  --verbose
(cram310) syt@zoulixin-4090D-48G:~/project/Cram$ python draft.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct   --n 100   --max_new_tokens 64   --temperature 0.0   --save_preds preds_nq_llama8b_inst.jsonl   --verbose
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.20s/it]
[10/100] EM/F1(no-ctx)=0/0.1  EM/F1(+ctx)=0/0.0
[20/100] EM/F1(no-ctx)=0/0.3  EM/F1(+ctx)=0/0.7
[30/100] EM/F1(no-ctx)=0/0.2  EM/F1(+ctx)=1/1.0
[40/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0
[50/100] EM/F1(no-ctx)=0/0.1  EM/F1(+ctx)=0/0.7
[60/100] EM/F1(no-ctx)=0/0.2  EM/F1(+ctx)=0/0.3
[70/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.2
[80/100] EM/F1(no-ctx)=0/0.7  EM/F1(+ctx)=1/1.0
[90/100] EM/F1(no-ctx)=0/0.4  EM/F1(+ctx)=0/0.4
[100/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.2

=== Final ===
(1) No context     : EM=6.0  F1=21.4
(2) With gold ctx  : EM=23.0  F1=45.7
Saved predictions to: preds_nq_llama8b_inst.jsonl



2.  python draft.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b \
  --n 100 \
  --max_new_tokens 64 \
  --temperature 0.0 \
  --save_preds preds_nq_llama8b.jsonl \
  --verbose
(cram310) syt@zoulixin-4090D-48G:~/project/Cram$ python draft.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b   --n 100   --max_new_tokens 64   --temperature 0.0   --save_preds preds_nq_llama8b.jsonl   --verbose
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.20s/it]
[10/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[20/100] EM/F1(no-ctx)=0/0.9  EM/F1(+ctx)=0/0.0
[30/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[40/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[50/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[60/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.3
[70/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[80/100] EM/F1(no-ctx)=0/0.2  EM/F1(+ctx)=0/0.2
[90/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[100/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0

=== Final ===
(1) No context     : EM=0.0  F1=8.5
(2) With gold ctx  : EM=5.0  F1=10.8
Saved predictions to: preds_nq_llama8b.jsonl



# 建议国内先切镜像
export HF_ENDPOINT=https://hf-mirror.com

pip install -U datasets transformers accelerate torch

python draft.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b \
  --n 100 \
  --split test \
  --max_new_tokens 16 \
  --temperature 0.0 \
  --csv_out preds_nq_llama8b.csv \
  --save_preds preds_nq_llama8b.jsonl \
  --verbose
(cram310) syt@zoulixin-4090D-48G:~/project/Cram$ python draft.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b   --n 100   --split test   --max_new_tokens 16   --temperature 0.0   --csv_out preds_nq_llama8b.csv   --save_preds preds_nq_llama8b.jsonl   --verbose
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.21s/it]
[10/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[20/100] EM/F1(no-ctx)=0/0.4  EM/F1(+ctx)=0/0.0
[30/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[40/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[50/100] EM/F1(no-ctx)=0/0.4  EM/F1(+ctx)=0/0.0
[60/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.7
[70/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[80/100] EM/F1(no-ctx)=0/0.3  EM/F1(+ctx)=0/0.0
[90/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[100/100] EM/F1(no-ctx)=0/0.3  EM/F1(+ctx)=0/0.0

=== Final ===
(1) No context     : EM=1.0  F1=8.2
(2) With gold ctx  : EM=4.0  F1=10.7
Saving CSV -> preds_nq_llama8b.csv
Saved JSONL -> preds_nq_llama8b.jsonl




python draft.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct \
  --n 100 \
  --split test \
  --max_new_tokens 16 \
  --temperature 0.0 \
  --csv_out preds_nq_llama8binstr.csv \
  --save_preds preds_nq_llama8binstr.jsonl \
  --verbose

(cram310) syt@zoulixin-4090D-48G:~/project/Cram$ python draft.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct \
  --n 100 \
  --split test \
  --max_new_tokens 16 \
  --temperature 0.0 \
  --csv_out preds_nq_llama8binstr.csv \
  --save_preds preds_nq_llama8binstr.jsonl \
  --verbose

Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.20s/it]
[10/100] EM/F1(no-ctx)=0/0.4  EM/F1(+ctx)=0/0.7
[20/100] EM/F1(no-ctx)=0/0.8  EM/F1(+ctx)=1/1.0
[30/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[40/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[50/100] EM/F1(no-ctx)=1/1.0  EM/F1(+ctx)=0/0.0
[60/100] EM/F1(no-ctx)=1/1.0  EM/F1(+ctx)=1/1.0
[70/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0
[80/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0
[90/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0
[100/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.5

=== Final ===
(1) No context     : EM=15.0  F1=27.3
(2) With gold ctx  : EM=44.0  F1=57.7
Saving CSV -> preds_nq_llama8binstr.csv
Saved JSONL -> preds_nq_llama8binstr.jsonl
