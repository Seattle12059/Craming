
# 测试
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



# 8b模型 
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



# instruct模型 ①  ②情况

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


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_nq_memtoken.py \
  --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct \
  --n 100 \
  --split test \
  --temperature 0.0 \
  --max_new_tokens 16 \
  --csv_out preds_nq_llama8b_mem1.csv \
  --save_preds preds_nq_llama8b_mem1.jsonl \
  --device cuda \
  --verbose \
  --mem_len 1 \
  --mem_steps 1000 \
  --mem_lr 1e-2 --mem_wd 1e-2 --mem_beta1 0.9 --mem_beta2 0.9 \
  --mem_target_acc 1.0 --mem_log_every 50 \
  --max_ctx_tokens 1024 \
  --ctx_window keyword \
  --enable_gc

[mem-train] step 1/1000  loss=1.4052  recon_acc=71.71%
[mem-train] step 50/1000  loss=0.2805  recon_acc=95.39%
[mem-train] step 100/1000  loss=0.0254  recon_acc=99.34%
[mem-train] early stop at step 109 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.1510  recon_acc=43.15%
[mem-train] step 50/1000  loss=0.4587  recon_acc=86.99%
[mem-train] step 100/1000  loss=0.0570  recon_acc=99.32%
[mem-train] step 150/1000  loss=0.0609  recon_acc=99.32%
[mem-train] early stop at step 197 with recon_acc=100.00%
[mem-train] step 1/1000  loss=4.4038  recon_acc=39.39%
[mem-train] step 50/1000  loss=2.1885  recon_acc=57.58%
[mem-train] step 100/1000  loss=0.1498  recon_acc=98.48%
[mem-train] step 150/1000  loss=0.1154  recon_acc=98.48%
[mem-train] early stop at step 196 with recon_acc=100.00%
[mem-train] step 1/1000  loss=9.6845  recon_acc=12.50%
[mem-train] early stop at step 19 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.5277  recon_acc=72.00%
[mem-train] step 50/1000  loss=1.2443  recon_acc=76.44%
[mem-train] step 100/1000  loss=0.3266  recon_acc=95.56%
[mem-train] step 150/1000  loss=0.0291  recon_acc=99.56%
[mem-train] early stop at step 180 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.3131  recon_acc=18.18%
[mem-train] step 50/1000  loss=0.4319  recon_acc=90.91%
[mem-train] step 100/1000  loss=0.3604  recon_acc=90.91%
[mem-train] early stop at step 115 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.3758  recon_acc=59.48%
[mem-train] step 50/1000  loss=0.7987  recon_acc=81.03%
[mem-train] step 100/1000  loss=0.0602  recon_acc=99.14%
[mem-train] early stop at step 126 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.6166  recon_acc=10.00%
[mem-train] early stop at step 27 with recon_acc=100.00%
[mem-train] step 1/1000  loss=4.8388  recon_acc=9.38%
[mem-train] step 50/1000  loss=0.7869  recon_acc=80.21%
[mem-train] early stop at step 72 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.1672  recon_acc=59.72%
[mem-train] step 50/1000  loss=1.4763  recon_acc=68.21%
[mem-train] step 100/1000  loss=0.7072  recon_acc=85.59%
[mem-train] step 150/1000  loss=0.3025  recon_acc=94.72%
[mem-train] step 200/1000  loss=0.0617  recon_acc=99.74%
[mem-train] step 250/1000  loss=0.0271  recon_acc=99.87%
[mem-train] step 300/1000  loss=0.0254  recon_acc=99.87%
[mem-train] step 350/1000  loss=0.0671  recon_acc=98.84%
[mem-train] step 400/1000  loss=0.0148  recon_acc=99.87%
[mem-train] step 450/1000  loss=0.0285  recon_acc=99.87%
[mem-train] step 500/1000  loss=0.0128  recon_acc=99.87%
[mem-train] step 550/1000  loss=0.0181  recon_acc=99.87%
[mem-train] step 600/1000  loss=0.0118  recon_acc=99.87%
[mem-train] step 650/1000  loss=0.0164  recon_acc=99.87%
[mem-train] step 700/1000  loss=0.0113  recon_acc=99.87%
[mem-train] step 750/1000  loss=0.0134  recon_acc=99.87%
[mem-train] step 800/1000  loss=0.0109  recon_acc=99.87%
[mem-train] step 850/1000  loss=0.0138  recon_acc=99.87%
[mem-train] step 900/1000  loss=0.0106  recon_acc=99.87%
[mem-train] step 950/1000  loss=0.0160  recon_acc=99.87%
[mem-train] step 1000/1000  loss=0.0105  recon_acc=99.87%
[10/100] EM/F1(no-ctx)=0/0.4  EM/F1(+ctx)=0/0.7  EM/F1(+1mem)=0/0.0  recon=99.9%
[mem-train] step 1/1000  loss=2.9792  recon_acc=47.86%
[mem-train] step 50/1000  loss=0.2777  recon_acc=96.58%
[mem-train] step 100/1000  loss=0.0391  recon_acc=99.15%
[mem-train] early stop at step 113 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.9697  recon_acc=63.05%
[mem-train] step 50/1000  loss=0.9499  recon_acc=76.71%
[mem-train] step 100/1000  loss=0.0607  recon_acc=99.20%
[mem-train] early stop at step 128 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.7297  recon_acc=40.00%
[mem-train] step 50/1000  loss=0.4397  recon_acc=90.00%
[mem-train] early stop at step 59 with recon_acc=100.00%
[mem-train] step 1/1000  loss=8.2013  recon_acc=39.37%
[mem-train] step 50/1000  loss=1.1439  recon_acc=74.02%
[mem-train] step 100/1000  loss=0.0913  recon_acc=99.21%
[mem-train] step 150/1000  loss=0.0794  recon_acc=99.21%
[mem-train] step 200/1000  loss=0.0784  recon_acc=99.21%
[mem-train] step 250/1000  loss=0.0680  recon_acc=99.21%
[mem-train] step 300/1000  loss=0.0360  recon_acc=100.00%
[mem-train] early stop at step 300 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.9710  recon_acc=57.66%
[mem-train] step 50/1000  loss=0.7912  recon_acc=81.75%
[mem-train] step 100/1000  loss=0.0708  recon_acc=99.27%
[mem-train] step 150/1000  loss=0.0622  recon_acc=99.27%
[mem-train] step 200/1000  loss=0.0595  recon_acc=99.27%
[mem-train] step 250/1000  loss=0.0448  recon_acc=99.27%
[mem-train] early stop at step 281 with recon_acc=100.00%
[mem-train] step 1/1000  loss=9.2993  recon_acc=16.10%
[mem-train] step 50/1000  loss=0.7261  recon_acc=85.85%
[mem-train] step 100/1000  loss=0.1736  recon_acc=98.05%
[mem-train] step 150/1000  loss=0.0395  recon_acc=99.51%
[mem-train] step 200/1000  loss=0.0348  recon_acc=99.51%
[mem-train] step 250/1000  loss=0.0335  recon_acc=99.51%
[mem-train] step 300/1000  loss=0.0291  recon_acc=99.51%
[mem-train] step 350/1000  loss=0.0316  recon_acc=99.51%
[mem-train] step 400/1000  loss=0.0275  recon_acc=99.51%
[mem-train] early stop at step 409 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.1092  recon_acc=57.65%
[mem-train] step 50/1000  loss=1.1641  recon_acc=72.94%
[mem-train] step 100/1000  loss=0.0605  recon_acc=98.82%
[mem-train] step 150/1000  loss=0.0376  recon_acc=98.82%
[mem-train] early stop at step 157 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.8936  recon_acc=47.87%
[mem-train] step 50/1000  loss=0.6649  recon_acc=87.68%
[mem-train] step 100/1000  loss=0.0230  recon_acc=99.53%
[mem-train] early stop at step 124 with recon_acc=100.00%
[mem-train] step 1/1000  loss=8.6055  recon_acc=4.23%
[mem-train] step 50/1000  loss=3.9010  recon_acc=38.03%
[mem-train] step 100/1000  loss=1.1441  recon_acc=74.65%
[mem-train] step 150/1000  loss=0.2315  recon_acc=97.18%
[mem-train] early stop at step 158 with recon_acc=100.00%
[mem-train] step 1/1000  loss=8.0575  recon_acc=64.10%
[mem-train] step 50/1000  loss=0.8234  recon_acc=87.18%
[mem-train] step 100/1000  loss=0.1554  recon_acc=97.44%
[mem-train] early stop at step 132 with recon_acc=100.00%
[20/100] EM/F1(no-ctx)=0/0.8  EM/F1(+ctx)=1/1.0  EM/F1(+1mem)=0/0.0  recon=100.0%
[mem-train] step 1/1000  loss=6.8437  recon_acc=54.61%
[mem-train] step 50/1000  loss=2.0238  recon_acc=56.74%
[mem-train] step 100/1000  loss=0.8563  recon_acc=79.43%
[mem-train] step 150/1000  loss=0.0608  recon_acc=99.29%
[mem-train] early stop at step 188 with recon_acc=100.00%
[mem-train] step 1/1000  loss=8.4872  recon_acc=11.11%
[mem-train] early stop at step 33 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.2219  recon_acc=1.51%
[mem-train] step 50/1000  loss=4.3348  recon_acc=24.91%
[mem-train] step 100/1000  loss=2.5251  recon_acc=44.53%
[mem-train] step 150/1000  loss=1.5154  recon_acc=68.30%
[mem-train] step 200/1000  loss=0.8279  recon_acc=83.40%
[mem-train] step 250/1000  loss=0.4543  recon_acc=94.34%
[mem-train] step 300/1000  loss=0.2759  recon_acc=95.85%
[mem-train] step 350/1000  loss=0.2406  recon_acc=98.11%
[mem-train] early stop at step 362 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.4389  recon_acc=79.20%
[mem-train] step 50/1000  loss=0.7407  recon_acc=88.00%
[mem-train] step 100/1000  loss=0.0662  recon_acc=99.20%
[mem-train] early stop at step 136 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.0712  recon_acc=77.86%
[mem-train] step 50/1000  loss=0.8567  recon_acc=82.50%
[mem-train] step 100/1000  loss=0.0380  recon_acc=99.64%
[mem-train] step 150/1000  loss=0.0283  recon_acc=99.64%
[mem-train] step 200/1000  loss=0.0173  recon_acc=100.00%
[mem-train] early stop at step 200 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.1682  recon_acc=62.60%
[mem-train] step 50/1000  loss=1.8409  recon_acc=64.12%
[mem-train] step 100/1000  loss=0.8411  recon_acc=84.73%
[mem-train] step 150/1000  loss=0.0699  recon_acc=99.24%
[mem-train] step 200/1000  loss=0.0594  recon_acc=99.24%
[mem-train] step 250/1000  loss=0.0560  recon_acc=99.24%
[mem-train] early stop at step 287 with recon_acc=100.00%
[mem-train] step 1/1000  loss=4.7220  recon_acc=64.52%
[mem-train] step 50/1000  loss=1.1520  recon_acc=77.42%
[mem-train] step 100/1000  loss=0.1076  recon_acc=98.92%
[mem-train] step 150/1000  loss=0.0725  recon_acc=98.92%
[mem-train] early stop at step 172 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.5166  recon_acc=3.23%
[mem-train] step 50/1000  loss=1.0831  recon_acc=83.87%
[mem-train] early stop at step 92 with recon_acc=100.00%
[mem-train] step 1/1000  loss=1.6726  recon_acc=68.24%
[mem-train] step 50/1000  loss=0.3285  recon_acc=94.71%
[mem-train] step 100/1000  loss=0.0386  recon_acc=99.41%
[mem-train] step 150/1000  loss=0.0364  recon_acc=99.41%
[mem-train] step 200/1000  loss=0.0177  recon_acc=99.41%
[mem-train] early stop at step 220 with recon_acc=100.00%
[mem-train] step 1/1000  loss=1.9228  recon_acc=58.85%
[mem-train] step 50/1000  loss=0.3335  recon_acc=89.58%
[mem-train] step 100/1000  loss=0.0229  recon_acc=99.48%
[mem-train] step 150/1000  loss=0.0174  recon_acc=99.48%
[mem-train] step 200/1000  loss=0.0181  recon_acc=99.48%
[mem-train] early stop at step 247 with recon_acc=100.00%
[30/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0  EM/F1(+1mem)=0/0.0  recon=100.0%
[mem-train] step 1/1000  loss=3.3559  recon_acc=49.02%
[mem-train] step 50/1000  loss=1.9079  recon_acc=55.56%
[mem-train] step 100/1000  loss=0.5042  recon_acc=91.50%
[mem-train] step 150/1000  loss=0.0643  recon_acc=99.35%
[mem-train] step 200/1000  loss=0.0536  recon_acc=99.35%
[mem-train] step 250/1000  loss=0.0541  recon_acc=99.35%
[mem-train] step 300/1000  loss=0.0522  recon_acc=99.35%
[mem-train] step 350/1000  loss=0.0511  recon_acc=99.35%
[mem-train] step 400/1000  loss=0.0469  recon_acc=99.35%
[mem-train] step 450/1000  loss=0.1499  recon_acc=99.35%
[mem-train] step 500/1000  loss=0.0450  recon_acc=99.35%
[mem-train] step 550/1000  loss=0.0401  recon_acc=99.35%
[mem-train] step 600/1000  loss=0.0412  recon_acc=99.35%
[mem-train] step 650/1000  loss=0.0339  recon_acc=99.35%
[mem-train] step 700/1000  loss=0.0359  recon_acc=99.35%
[mem-train] step 750/1000  loss=0.0361  recon_acc=99.35%
[mem-train] early stop at step 776 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.3817  recon_acc=54.90%
[mem-train] step 50/1000  loss=1.4429  recon_acc=64.71%
[mem-train] step 100/1000  loss=0.0794  recon_acc=99.02%
[mem-train] step 150/1000  loss=0.0570  recon_acc=99.02%
[mem-train] step 200/1000  loss=0.0562  recon_acc=99.02%
[mem-train] early stop at step 240 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.2713  recon_acc=46.60%
[mem-train] step 50/1000  loss=0.2417  recon_acc=97.09%
[mem-train] step 100/1000  loss=0.0540  recon_acc=99.03%
[mem-train] early stop at step 119 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.4715  recon_acc=57.81%
[mem-train] step 50/1000  loss=0.1662  recon_acc=97.66%
[mem-train] step 100/1000  loss=0.0314  recon_acc=99.22%
[mem-train] step 150/1000  loss=0.0403  recon_acc=97.66%
[mem-train] step 200/1000  loss=0.0252  recon_acc=99.22%
[mem-train] early stop at step 226 with recon_acc=100.00%
[mem-train] step 1/1000  loss=1.5140  recon_acc=64.89%
[mem-train] step 50/1000  loss=0.1327  recon_acc=96.00%
[mem-train] step 100/1000  loss=0.0231  recon_acc=99.56%
[mem-train] step 150/1000  loss=0.0159  recon_acc=99.56%
[mem-train] step 200/1000  loss=0.0153  recon_acc=99.56%
[mem-train] early stop at step 249 with recon_acc=100.00%
[mem-trunc] full=8211  keep=1024  range=(611,1635) mode=keyword
[mem-train] step 1/1000  loss=2.8120  recon_acc=44.73%
[mem-train] step 50/1000  loss=2.3858  recon_acc=49.41%
[mem-train] step 100/1000  loss=1.4861  recon_acc=66.80%
[mem-train] step 150/1000  loss=0.7503  recon_acc=83.40%
[mem-train] step 200/1000  loss=0.4488  recon_acc=90.23%
[mem-train] step 250/1000  loss=0.2760  recon_acc=96.39%
[mem-train] step 300/1000  loss=0.1684  recon_acc=98.44%
[mem-train] step 350/1000  loss=0.1131  recon_acc=99.02%
[mem-train] step 400/1000  loss=0.1973  recon_acc=94.82%
[mem-train] step 450/1000  loss=0.0315  recon_acc=99.80%
[mem-train] step 500/1000  loss=0.0611  recon_acc=99.80%
[mem-train] step 550/1000  loss=0.0201  recon_acc=99.90%
[mem-train] step 600/1000  loss=0.0303  recon_acc=99.80%
[mem-train] step 650/1000  loss=0.0159  recon_acc=99.90%
[mem-train] step 700/1000  loss=0.0189  recon_acc=99.90%
[mem-train] step 750/1000  loss=0.2686  recon_acc=96.58%
[mem-train] step 800/1000  loss=0.0159  recon_acc=99.90%
[mem-train] step 850/1000  loss=0.2796  recon_acc=97.95%
[mem-train] step 900/1000  loss=0.0157  recon_acc=99.90%
[mem-train] step 950/1000  loss=0.1001  recon_acc=96.00%
[mem-train] step 1000/1000  loss=0.0124  recon_acc=99.90%
[mem-train] step 1/1000  loss=7.5648  recon_acc=44.78%
[mem-train] step 50/1000  loss=1.7568  recon_acc=61.19%
[mem-train] step 100/1000  loss=0.0812  recon_acc=99.25%
[mem-train] step 150/1000  loss=0.0579  recon_acc=99.25%
[mem-train] step 200/1000  loss=0.0565  recon_acc=99.25%
[mem-train] step 250/1000  loss=0.0437  recon_acc=99.25%
[mem-train] early stop at step 261 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.0891  recon_acc=57.14%
[mem-train] step 50/1000  loss=0.2269  recon_acc=97.25%
[mem-train] step 100/1000  loss=0.0440  recon_acc=99.45%
[mem-train] step 150/1000  loss=0.0284  recon_acc=99.45%
[mem-train] early stop at step 162 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.4770  recon_acc=74.12%
[mem-train] step 50/1000  loss=0.9668  recon_acc=77.65%
[mem-train] step 100/1000  loss=0.0642  recon_acc=99.41%
[mem-train] step 150/1000  loss=0.0230  recon_acc=99.41%
[mem-train] step 200/1000  loss=0.0213  recon_acc=99.41%
[mem-train] early stop at step 217 with recon_acc=100.00%
[mem-train] step 1/1000  loss=1.3323  recon_acc=77.93%
[mem-train] step 50/1000  loss=0.4612  recon_acc=91.03%
[mem-train] step 100/1000  loss=0.0282  recon_acc=99.31%
[mem-train] early stop at step 149 with recon_acc=100.00%
[40/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0  EM/F1(+1mem)=0/0.0  recon=100.0%
[mem-train] step 1/1000  loss=6.6915  recon_acc=49.44%
[mem-train] step 50/1000  loss=1.1730  recon_acc=72.12%
[mem-train] step 100/1000  loss=0.0378  recon_acc=99.63%
[mem-train] step 150/1000  loss=0.0299  recon_acc=99.63%
[mem-train] step 200/1000  loss=0.0332  recon_acc=99.63%
[mem-train] step 250/1000  loss=0.0285  recon_acc=99.63%
[mem-train] step 300/1000  loss=0.0271  recon_acc=98.88%
[mem-train] step 350/1000  loss=0.0259  recon_acc=99.63%
[mem-train] step 400/1000  loss=0.0356  recon_acc=99.63%
[mem-train] step 450/1000  loss=0.0232  recon_acc=99.63%
[mem-train] step 500/1000  loss=0.0227  recon_acc=99.63%
[mem-train] step 550/1000  loss=0.0152  recon_acc=99.63%
[mem-train] early stop at step 552 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.0032  recon_acc=56.87%
[mem-train] step 50/1000  loss=1.0772  recon_acc=80.15%
[mem-train] step 100/1000  loss=0.0753  recon_acc=99.24%
[mem-train] early stop at step 116 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.0832  recon_acc=63.16%
[mem-train] step 50/1000  loss=0.4389  recon_acc=88.82%
[mem-train] step 100/1000  loss=0.0345  recon_acc=99.34%
[mem-train] step 150/1000  loss=0.0291  recon_acc=99.34%
[mem-train] step 200/1000  loss=0.0230  recon_acc=99.34%
[mem-train] early stop at step 219 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.0051  recon_acc=14.81%
[mem-train] step 50/1000  loss=1.5491  recon_acc=68.52%
[mem-train] early stop at step 77 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.3108  recon_acc=16.33%
[mem-train] step 50/1000  loss=1.2213  recon_acc=74.15%
[mem-train] early stop at step 79 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.0001  recon_acc=70.05%
[mem-train] step 50/1000  loss=1.1223  recon_acc=76.04%
[mem-train] step 100/1000  loss=0.0443  recon_acc=99.54%
[mem-train] step 150/1000  loss=0.0351  recon_acc=99.54%
[mem-train] step 200/1000  loss=0.0256  recon_acc=99.54%
[mem-train] early stop at step 232 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.5918  recon_acc=69.17%
[mem-train] step 50/1000  loss=0.7713  recon_acc=80.42%
[mem-train] step 100/1000  loss=0.0322  recon_acc=99.58%
[mem-train] step 150/1000  loss=0.0150  recon_acc=99.58%
[mem-train] early stop at step 174 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.3522  recon_acc=6.88%
[mem-train] step 50/1000  loss=1.9131  recon_acc=60.62%
[mem-train] step 100/1000  loss=0.2492  recon_acc=97.50%
[mem-train] early stop at step 119 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.5634  recon_acc=65.22%
[mem-train] step 50/1000  loss=1.0004  recon_acc=82.61%
[mem-train] step 100/1000  loss=0.0981  recon_acc=98.55%
[mem-train] early stop at step 131 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.9444  recon_acc=55.04%
[mem-train] step 50/1000  loss=0.5558  recon_acc=87.60%
[mem-train] step 100/1000  loss=0.0579  recon_acc=99.22%
[mem-train] step 150/1000  loss=0.0606  recon_acc=99.22%
[mem-train] step 200/1000  loss=0.0446  recon_acc=99.22%
[mem-train] early stop at step 226 with recon_acc=100.00%
[50/100] EM/F1(no-ctx)=1/1.0  EM/F1(+ctx)=0/0.0  EM/F1(+1mem)=1/1.0  recon=100.0%
[mem-train] step 1/1000  loss=2.7005  recon_acc=60.00%
[mem-train] step 50/1000  loss=0.5988  recon_acc=87.50%
[mem-train] step 100/1000  loss=0.0869  recon_acc=97.50%
[mem-train] early stop at step 129 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.3912  recon_acc=54.39%
[mem-train] step 50/1000  loss=0.2307  recon_acc=97.37%
[mem-train] step 100/1000  loss=0.0317  recon_acc=99.12%
[mem-train] early stop at step 102 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.6694  recon_acc=6.94%
[mem-train] step 50/1000  loss=2.3546  recon_acc=45.83%
[mem-train] step 100/1000  loss=0.1343  recon_acc=98.61%
[mem-train] step 150/1000  loss=0.0855  recon_acc=98.61%
[mem-train] early stop at step 187 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.0498  recon_acc=42.73%
[mem-train] step 50/1000  loss=2.4427  recon_acc=42.95%
[mem-train] step 100/1000  loss=1.9831  recon_acc=54.63%
[mem-train] step 150/1000  loss=0.6062  recon_acc=87.67%
[mem-train] step 200/1000  loss=0.0348  recon_acc=99.78%
[mem-train] step 250/1000  loss=0.0238  recon_acc=98.90%
[mem-train] step 300/1000  loss=0.0227  recon_acc=99.78%
[mem-train] step 350/1000  loss=0.0329  recon_acc=99.78%
[mem-train] step 400/1000  loss=0.0204  recon_acc=99.78%
[mem-train] step 450/1000  loss=0.0341  recon_acc=98.02%
[mem-train] step 500/1000  loss=0.0187  recon_acc=99.78%
[mem-train] step 550/1000  loss=0.0624  recon_acc=82.82%
[mem-train] step 600/1000  loss=0.0174  recon_acc=99.78%
[mem-train] step 650/1000  loss=0.0130  recon_acc=99.78%
[mem-train] step 700/1000  loss=0.0155  recon_acc=99.78%
[mem-train] step 750/1000  loss=0.0104  recon_acc=95.37%
[mem-train] step 800/1000  loss=0.0139  recon_acc=99.78%
[mem-train] step 850/1000  loss=0.0084  recon_acc=99.78%
[mem-train] early stop at step 860 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.9446  recon_acc=54.29%
[mem-train] step 50/1000  loss=1.6536  recon_acc=66.12%
[mem-train] step 100/1000  loss=0.3503  recon_acc=93.47%
[mem-train] step 150/1000  loss=0.0237  recon_acc=99.59%
[mem-train] step 200/1000  loss=0.1489  recon_acc=99.59%
[mem-train] step 250/1000  loss=0.0157  recon_acc=99.59%
[mem-train] early stop at step 296 with recon_acc=100.00%
[mem-trunc] full=3479  keep=1024  range=(2455,3479) mode=keyword
[mem-train] step 1/1000  loss=1.6067  recon_acc=69.73%
[mem-train] step 50/1000  loss=1.3345  recon_acc=73.63%
[mem-train] step 100/1000  loss=0.7488  recon_acc=85.45%
[mem-train] step 150/1000  loss=0.3607  recon_acc=94.14%
[mem-train] step 200/1000  loss=0.1494  recon_acc=97.66%
[mem-train] step 250/1000  loss=0.0821  recon_acc=99.71%
[mem-train] step 300/1000  loss=0.0178  recon_acc=99.90%
[mem-train] step 350/1000  loss=0.0185  recon_acc=99.90%
[mem-train] step 400/1000  loss=0.0137  recon_acc=99.90%
[mem-train] step 450/1000  loss=0.0149  recon_acc=99.90%
[mem-train] step 500/1000  loss=0.0131  recon_acc=99.90%
[mem-train] step 550/1000  loss=0.0145  recon_acc=99.90%
[mem-train] step 600/1000  loss=0.0128  recon_acc=99.90%
[mem-train] step 650/1000  loss=0.0137  recon_acc=99.90%
[mem-train] step 700/1000  loss=0.0125  recon_acc=99.90%
[mem-train] step 750/1000  loss=0.0141  recon_acc=99.90%
[mem-train] step 800/1000  loss=0.0123  recon_acc=99.90%
[mem-train] step 850/1000  loss=0.0138  recon_acc=99.90%
[mem-train] step 900/1000  loss=0.0121  recon_acc=99.90%
[mem-train] step 950/1000  loss=0.0130  recon_acc=99.90%
[mem-train] step 1000/1000  loss=0.0118  recon_acc=99.90%
[mem-train] step 1/1000  loss=6.9054  recon_acc=50.80%
[mem-train] step 50/1000  loss=1.1119  recon_acc=79.14%
[mem-train] step 100/1000  loss=0.0515  recon_acc=99.47%
[mem-train] step 150/1000  loss=0.0434  recon_acc=99.47%
[mem-train] step 200/1000  loss=0.0324  recon_acc=99.47%
[mem-train] step 250/1000  loss=0.0281  recon_acc=99.47%
[mem-train] step 300/1000  loss=0.0192  recon_acc=99.47%
[mem-train] early stop at step 314 with recon_acc=100.00%
[mem-train] step 1/1000  loss=1.9686  recon_acc=60.71%
[mem-train] step 50/1000  loss=0.1160  recon_acc=98.81%
[mem-train] step 100/1000  loss=0.0929  recon_acc=98.81%
[mem-train] early stop at step 132 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.9028  recon_acc=0.00%
[mem-train] step 50/1000  loss=3.3092  recon_acc=43.90%
[mem-train] early stop at step 89 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.6853  recon_acc=7.03%
[mem-train] step 50/1000  loss=1.5981  recon_acc=64.84%
[mem-train] step 100/1000  loss=0.0809  recon_acc=99.22%
[mem-train] step 150/1000  loss=0.0638  recon_acc=99.22%
[mem-train] step 200/1000  loss=0.0485  recon_acc=99.22%
[mem-train] early stop at step 224 with recon_acc=100.00%
[60/100] EM/F1(no-ctx)=1/1.0  EM/F1(+ctx)=1/1.0  EM/F1(+1mem)=1/1.0  recon=100.0%
[mem-train] step 1/1000  loss=7.8169  recon_acc=23.30%
[mem-train] step 50/1000  loss=2.1130  recon_acc=58.25%
[mem-train] step 100/1000  loss=1.3012  recon_acc=67.96%
[mem-train] step 150/1000  loss=0.8915  recon_acc=83.50%
[mem-train] step 200/1000  loss=0.0651  recon_acc=99.03%
[mem-train] early stop at step 232 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.4558  recon_acc=18.03%
[mem-train] step 50/1000  loss=0.2165  recon_acc=98.36%
[mem-train] early stop at step 54 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.5246  recon_acc=57.95%
[mem-train] step 50/1000  loss=0.5133  recon_acc=89.77%
[mem-train] step 100/1000  loss=0.0375  recon_acc=99.43%
[mem-train] early stop at step 143 with recon_acc=100.00%
[mem-train] step 1/1000  loss=0.9222  recon_acc=79.51%
[mem-train] step 50/1000  loss=0.0323  recon_acc=99.18%
[mem-train] step 100/1000  loss=0.0254  recon_acc=99.18%
[mem-train] early stop at step 105 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.8378  recon_acc=15.19%
[mem-train] step 50/1000  loss=0.7653  recon_acc=83.54%
[mem-train] step 100/1000  loss=0.1205  recon_acc=98.73%
[mem-train] early stop at step 149 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.7138  recon_acc=58.29%
[mem-train] step 50/1000  loss=0.0877  recon_acc=99.05%
[mem-train] step 100/1000  loss=0.0503  recon_acc=99.53%
[mem-train] early stop at step 144 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.7509  recon_acc=44.71%
[mem-train] step 50/1000  loss=0.6194  recon_acc=87.84%
[mem-train] step 100/1000  loss=0.0328  recon_acc=99.61%
[mem-train] step 150/1000  loss=0.1384  recon_acc=94.51%
[mem-train] step 200/1000  loss=0.0280  recon_acc=99.61%
[mem-train] step 250/1000  loss=0.0191  recon_acc=99.61%
[mem-train] step 300/1000  loss=0.0179  recon_acc=99.61%
[mem-train] early stop at step 315 with recon_acc=100.00%
[mem-train] step 1/1000  loss=8.6586  recon_acc=6.25%
[mem-train] step 50/1000  loss=0.3191  recon_acc=97.92%
[mem-train] early stop at step 85 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.7108  recon_acc=45.16%
[mem-train] step 50/1000  loss=0.7913  recon_acc=84.68%
[mem-train] step 100/1000  loss=0.0870  recon_acc=99.19%
[mem-train] step 150/1000  loss=0.0680  recon_acc=99.19%
[mem-train] step 200/1000  loss=0.0623  recon_acc=99.19%
[mem-train] step 250/1000  loss=0.0377  recon_acc=99.19%
[mem-train] early stop at step 254 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.6053  recon_acc=58.33%
[mem-train] step 50/1000  loss=1.6803  recon_acc=62.96%
[mem-train] step 100/1000  loss=0.0712  recon_acc=99.07%
[mem-train] step 150/1000  loss=0.0338  recon_acc=99.07%
[mem-train] early stop at step 180 with recon_acc=100.00%
[70/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0  EM/F1(+1mem)=0/0.0  recon=100.0%
[mem-train] step 1/1000  loss=1.8459  recon_acc=17.78%
[mem-train] step 50/1000  loss=0.5560  recon_acc=91.11%
[mem-train] early stop at step 60 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.7610  recon_acc=18.37%
[mem-train] step 50/1000  loss=0.8076  recon_acc=79.59%
[mem-train] step 100/1000  loss=0.0505  recon_acc=98.98%
[mem-train] early stop at step 116 with recon_acc=100.00%
[mem-train] step 1/1000  loss=4.7636  recon_acc=36.00%
[mem-train] step 50/1000  loss=1.0520  recon_acc=80.00%
[mem-train] step 100/1000  loss=0.1989  recon_acc=96.00%
[mem-train] step 150/1000  loss=0.1588  recon_acc=96.00%
[mem-train] step 200/1000  loss=0.1296  recon_acc=96.00%
[mem-train] early stop at step 209 with recon_acc=100.00%
[mem-train] step 1/1000  loss=0.6023  recon_acc=85.37%
[mem-train] step 50/1000  loss=0.2291  recon_acc=93.74%
[mem-train] step 100/1000  loss=0.0233  recon_acc=99.70%
[mem-train] step 150/1000  loss=0.0093  recon_acc=99.90%
[mem-train] step 200/1000  loss=0.0103  recon_acc=99.90%
[mem-train] step 250/1000  loss=0.0079  recon_acc=99.90%
[mem-train] step 300/1000  loss=0.0110  recon_acc=99.90%
[mem-train] step 350/1000  loss=0.0070  recon_acc=99.90%
[mem-train] step 400/1000  loss=0.0072  recon_acc=99.90%
[mem-train] step 450/1000  loss=0.0059  recon_acc=99.90%
[mem-train] early stop at step 471 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.7536  recon_acc=47.83%
[mem-train] step 50/1000  loss=0.6219  recon_acc=89.13%
[mem-train] early stop at step 99 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.3968  recon_acc=45.45%
[mem-train] step 50/1000  loss=1.9995  recon_acc=50.91%
[mem-train] step 100/1000  loss=0.1010  recon_acc=98.18%
[mem-train] step 150/1000  loss=0.0569  recon_acc=98.18%
[mem-train] early stop at step 155 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.4842  recon_acc=7.69%
[mem-train] step 50/1000  loss=6.5216  recon_acc=7.69%
[mem-train] step 100/1000  loss=0.7334  recon_acc=89.01%
[mem-train] step 150/1000  loss=0.1130  recon_acc=98.90%
[mem-train] step 200/1000  loss=0.1015  recon_acc=98.90%
[mem-train] early stop at step 223 with recon_acc=100.00%
[mem-train] step 1/1000  loss=1.0295  recon_acc=79.06%
[mem-train] step 50/1000  loss=0.3198  recon_acc=91.36%
[mem-train] step 100/1000  loss=0.0302  recon_acc=99.74%
[mem-train] step 150/1000  loss=0.0194  recon_acc=99.74%
[mem-train] step 200/1000  loss=0.0200  recon_acc=99.74%
[mem-train] step 250/1000  loss=0.0167  recon_acc=99.21%
[mem-train] step 300/1000  loss=0.0176  recon_acc=99.74%
[mem-train] step 350/1000  loss=0.0161  recon_acc=99.74%
[mem-train] early stop at step 377 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.5984  recon_acc=62.07%
[mem-train] step 50/1000  loss=0.1098  recon_acc=98.85%
[mem-train] step 100/1000  loss=0.0711  recon_acc=98.85%
[mem-train] early stop at step 135 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.7956  recon_acc=56.18%
[mem-train] step 50/1000  loss=0.2201  recon_acc=97.75%
[mem-train] early stop at step 96 with recon_acc=100.00%
[80/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0  EM/F1(+1mem)=0/0.4  recon=100.0%
[mem-train] step 1/1000  loss=5.4256  recon_acc=66.67%
[mem-train] step 50/1000  loss=0.9907  recon_acc=75.00%
[mem-train] step 100/1000  loss=0.0546  recon_acc=99.36%
[mem-train] step 150/1000  loss=0.0231  recon_acc=99.36%
[mem-train] early stop at step 182 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.4704  recon_acc=69.91%
[mem-train] step 50/1000  loss=0.0785  recon_acc=99.12%
[mem-train] step 100/1000  loss=0.0546  recon_acc=99.12%
[mem-train] step 150/1000  loss=0.0490  recon_acc=99.12%
[mem-train] early stop at step 163 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.7767  recon_acc=46.54%
[mem-train] step 50/1000  loss=1.6139  recon_acc=59.45%
[mem-train] step 100/1000  loss=0.0470  recon_acc=99.54%
[mem-train] step 150/1000  loss=0.0214  recon_acc=99.54%
[mem-train] step 200/1000  loss=0.0270  recon_acc=99.54%
[mem-train] step 250/1000  loss=0.0191  recon_acc=99.54%
[mem-train] early stop at step 292 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.3263  recon_acc=65.38%
[mem-train] step 50/1000  loss=1.1329  recon_acc=79.81%
[mem-train] step 100/1000  loss=0.0733  recon_acc=99.04%
[mem-train] step 150/1000  loss=0.0307  recon_acc=100.00%
[mem-train] early stop at step 150 with recon_acc=100.00%
[mem-train] step 1/1000  loss=4.8158  recon_acc=13.04%
[mem-train] step 50/1000  loss=0.8703  recon_acc=86.96%
[mem-train] step 100/1000  loss=0.1120  recon_acc=98.55%
[mem-train] step 150/1000  loss=0.0944  recon_acc=98.55%
[mem-train] step 200/1000  loss=0.0798  recon_acc=98.55%
[mem-train] early stop at step 216 with recon_acc=100.00%
[mem-train] step 1/1000  loss=5.2173  recon_acc=44.82%
[mem-train] step 50/1000  loss=2.7066  recon_acc=44.82%
[mem-train] step 100/1000  loss=2.1288  recon_acc=55.36%
[mem-train] step 150/1000  loss=0.6374  recon_acc=87.14%
[mem-train] step 200/1000  loss=0.1143  recon_acc=88.39%
[mem-train] step 250/1000  loss=0.0469  recon_acc=99.64%
[mem-train] step 300/1000  loss=0.0370  recon_acc=99.82%
[mem-train] step 350/1000  loss=0.0186  recon_acc=99.82%
[mem-train] step 400/1000  loss=0.0283  recon_acc=99.82%
[mem-train] step 450/1000  loss=0.0177  recon_acc=99.82%
[mem-train] step 500/1000  loss=0.0714  recon_acc=99.46%
[mem-train] step 550/1000  loss=0.0177  recon_acc=99.82%
[mem-train] step 600/1000  loss=0.0317  recon_acc=96.79%
[mem-train] step 650/1000  loss=0.0176  recon_acc=99.82%
[mem-train] step 700/1000  loss=0.0162  recon_acc=99.82%
[mem-train] step 750/1000  loss=0.0171  recon_acc=99.82%
[mem-train] step 800/1000  loss=0.0159  recon_acc=99.82%
[mem-train] step 850/1000  loss=0.0174  recon_acc=99.82%
[mem-train] step 900/1000  loss=0.0158  recon_acc=99.82%
[mem-train] step 950/1000  loss=0.0196  recon_acc=99.82%
[mem-train] step 1000/1000  loss=0.0157  recon_acc=99.82%
[mem-train] step 1/1000  loss=1.8156  recon_acc=70.53%
[mem-train] step 50/1000  loss=0.0493  recon_acc=98.95%
[mem-train] step 100/1000  loss=0.0317  recon_acc=100.00%
[mem-train] early stop at step 100 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.8655  recon_acc=43.97%
[mem-train] step 50/1000  loss=1.6906  recon_acc=61.83%
[mem-train] step 100/1000  loss=0.1114  recon_acc=99.11%
[mem-train] step 150/1000  loss=0.0220  recon_acc=99.78%
[mem-train] step 200/1000  loss=0.0294  recon_acc=99.78%
[mem-train] step 250/1000  loss=0.0193  recon_acc=99.78%
[mem-train] step 300/1000  loss=0.1542  recon_acc=98.66%
[mem-train] step 350/1000  loss=0.0180  recon_acc=99.78%
[mem-train] step 400/1000  loss=0.0154  recon_acc=99.78%
[mem-train] step 450/1000  loss=0.0164  recon_acc=99.78%
[mem-train] step 500/1000  loss=0.0937  recon_acc=99.33%
[mem-train] step 550/1000  loss=0.0149  recon_acc=99.78%
[mem-train] step 600/1000  loss=0.0114  recon_acc=99.78%
[mem-train] step 650/1000  loss=0.0128  recon_acc=99.78%
[mem-train] early stop at step 676 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.8409  recon_acc=65.66%
[mem-train] step 50/1000  loss=1.3646  recon_acc=71.69%
[mem-train] step 100/1000  loss=0.0638  recon_acc=99.40%
[mem-train] step 150/1000  loss=0.0468  recon_acc=99.40%
[mem-train] early stop at step 169 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.6177  recon_acc=37.93%
[mem-train] step 50/1000  loss=0.9382  recon_acc=86.21%
[mem-train] step 100/1000  loss=0.1059  recon_acc=98.85%
[mem-train] step 150/1000  loss=0.0933  recon_acc=98.85%
[mem-train] step 200/1000  loss=0.0966  recon_acc=98.85%
[mem-train] step 250/1000  loss=0.0874  recon_acc=98.85%
[mem-train] step 300/1000  loss=0.0911  recon_acc=98.85%
[mem-train] step 350/1000  loss=0.0777  recon_acc=98.85%
[mem-train] early stop at step 387 with recon_acc=100.00%
[90/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0  EM/F1(+1mem)=0/0.0  recon=100.0%
[mem-train] step 1/1000  loss=1.9403  recon_acc=63.46%
[mem-train] step 50/1000  loss=0.1350  recon_acc=90.38%
[mem-train] step 100/1000  loss=0.0565  recon_acc=99.04%
[mem-train] early stop at step 118 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.3496  recon_acc=53.57%
[mem-train] step 50/1000  loss=0.9025  recon_acc=83.93%
[mem-train] early stop at step 59 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.2025  recon_acc=54.65%
[mem-train] step 50/1000  loss=0.3833  recon_acc=94.19%
[mem-train] step 100/1000  loss=0.0557  recon_acc=98.84%
[mem-train] early stop at step 139 with recon_acc=100.00%
[mem-train] step 1/1000  loss=2.2412  recon_acc=57.14%
[mem-train] step 50/1000  loss=0.1816  recon_acc=97.74%
[mem-train] step 100/1000  loss=0.0432  recon_acc=99.25%
[mem-train] early stop at step 137 with recon_acc=100.00%
[mem-train] step 1/1000  loss=3.9680  recon_acc=73.90%
[mem-train] step 50/1000  loss=1.0740  recon_acc=79.59%
[mem-train] step 100/1000  loss=0.0972  recon_acc=98.71%
[mem-train] step 150/1000  loss=0.0207  recon_acc=99.74%
[mem-train] step 200/1000  loss=0.0210  recon_acc=99.74%
[mem-train] step 250/1000  loss=0.0188  recon_acc=99.74%
[mem-train] step 300/1000  loss=0.0171  recon_acc=99.74%
[mem-train] step 350/1000  loss=0.0121  recon_acc=99.74%
[mem-train] early stop at step 356 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.5211  recon_acc=14.85%
[mem-train] step 50/1000  loss=1.1675  recon_acc=76.24%
[mem-train] step 100/1000  loss=0.1710  recon_acc=99.01%
[mem-train] early stop at step 105 with recon_acc=100.00%
[mem-train] step 1/1000  loss=1.1337  recon_acc=81.94%
[mem-train] step 50/1000  loss=0.1080  recon_acc=98.71%
[mem-train] step 100/1000  loss=0.0485  recon_acc=99.35%
[mem-train] step 150/1000  loss=0.0414  recon_acc=99.35%
[mem-train] early stop at step 173 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.5694  recon_acc=13.71%
[mem-train] step 50/1000  loss=1.8092  recon_acc=53.71%
[mem-train] step 100/1000  loss=0.1791  recon_acc=98.86%
[mem-train] early stop at step 137 with recon_acc=100.00%
[mem-train] step 1/1000  loss=7.3738  recon_acc=13.49%
[mem-train] step 50/1000  loss=1.6444  recon_acc=65.08%
[mem-train] early stop at step 93 with recon_acc=100.00%
[mem-train] step 1/1000  loss=6.0789  recon_acc=45.56%
[mem-train] step 50/1000  loss=0.8392  recon_acc=81.11%
[mem-train] step 100/1000  loss=0.0736  recon_acc=98.89%
[mem-train] step 150/1000  loss=0.0434  recon_acc=98.89%
[mem-train] step 200/1000  loss=0.0745  recon_acc=98.89%
[mem-train] step 250/1000  loss=0.0402  recon_acc=98.89%
[mem-train] early stop at step 296 with recon_acc=100.00%
[100/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.5  EM/F1(+1mem)=0/0.0  recon=100.0%

=== Final ===
(1) No context     : EM=15.0  F1=27.3
(2) With gold ctx  : EM=44.0  F1=57.7
(3) With 1 mem     : EM=12.0  F1=21.8  (mem QA，仅统计)
Saving CSV -> preds_nq_llama8b_mem1.csv
Saved JSONL -> preds_nq_llama8b_mem1.jsonl
