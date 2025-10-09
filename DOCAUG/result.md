
### 复杂任务 4mem 10样本无一幸免
(cram310L20) syt@zoulixin-L20:~/project/cram_L20/DOCAUG$ python -u train_nq_mem_joint_kd.py \
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
  --limit 10\s runs/nq_joint_preds_kd400step4mem_dif.jsonl \

`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.85s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.86s/it]
[mem-train] step 1/400 alpha=0.80 loss=22.5298 AE=0.8473 QA=8.0100 KD=20.2500 recon=75.87%
[mem-train] step 100/400 alpha=0.70 loss=13.0830 AE=0.9611 QA=2.0735 KD=11.7891 recon=76.73%
[mem-train] step 200/400 alpha=0.60 loss=4.3191 AE=0.9599 QA=0.0989 KD=3.7031 recon=76.30%
[mem-train] step 300/400 alpha=0.50 loss=1.4331 AE=0.8712 QA=0.0054 KD=0.9946 recon=77.89%
[mem-train] step 400/400 alpha=0.40 loss=1.1846 AE=0.9020 QA=0.0001 KD=0.8237 recon=77.60%
[mem-train] step 1/400 alpha=0.80 loss=21.2750 AE=0.8942 QA=6.8607 KD=19.1875 recon=56.50%
[mem-train] step 100/400 alpha=0.70 loss=3.1912 AE=1.8163 QA=0.0048 KD=1.9170 recon=70.38%
[mem-train] step 200/400 alpha=0.60 loss=1.9504 AE=1.6776 QA=0.0003 KD=0.9429 recon=71.24%
[mem-train] step 300/400 alpha=0.50 loss=2.7272 AE=1.5439 QA=0.0055 KD=1.9521 recon=72.25%
[mem-train] step 400/400 alpha=0.40 loss=1.3210 AE=1.5048 QA=0.0005 KD=0.7188 recon=71.68%
[mem-train] step 1/400 alpha=0.80 loss=24.9276 AE=1.8624 QA=4.4540 KD=22.5469 recon=42.69%
[mem-train] step 100/400 alpha=0.70 loss=9.4325 AE=2.3916 QA=0.2960 KD=7.6680 recon=64.61%
[mem-train] step 200/400 alpha=0.60 loss=3.8244 AE=2.1328 QA=0.0067 KD=2.5410 recon=65.91%
[mem-train] step 300/400 alpha=0.50 loss=3.1482 AE=2.0023 QA=0.0041 KD=2.1445 recon=66.88%
[mem-train] step 400/400 alpha=0.40 loss=2.7348 AE=1.8963 QA=0.0012 KD=1.9756 recon=67.21%
[mem-train] step 1/400 alpha=0.80 loss=36.1266 AE=2.8912 QA=6.5682 KD=32.5000 recon=35.77%
[mem-train] step 100/400 alpha=0.70 loss=5.0354 AE=0.6844 QA=0.1343 KD=4.5156 recon=85.98%
[mem-train] step 200/400 alpha=0.60 loss=3.5223 AE=0.6652 QA=0.0727 KD=3.0938 recon=86.41%
[mem-train] step 300/400 alpha=0.50 loss=4.3627 AE=0.6364 QA=0.1044 KD=3.9922 recon=86.55%
[mem-train] step 400/400 alpha=0.40 loss=1.6170 AE=0.6212 QA=0.0006 KD=1.3682 recon=86.84%
[mem-train] step 1/400 alpha=0.80 loss=24.7178 AE=1.2469 QA=4.6168 KD=22.7969 recon=82.76%
[mem-train] step 100/400 alpha=0.70 loss=2.0048 AE=0.7196 QA=0.0018 KD=1.5000 recon=83.97%
[mem-train] step 200/400 alpha=0.60 loss=1.3881 AE=0.5931 QA=0.0016 KD=1.0312 recon=84.14%
[mem-train] step 300/400 alpha=0.50 loss=1.0236 AE=0.4990 QA=0.0011 KD=0.7734 recon=86.72%
[mem-train] step 400/400 alpha=0.40 loss=0.9262 AE=0.4218 QA=0.0011 KD=0.7568 recon=89.14%
[mem-train] step 1/400 alpha=0.80 loss=7.9329 AE=1.0622 QA=0.4740 KD=6.9883 recon=87.10%
[mem-train] step 100/400 alpha=0.70 loss=3.1584 AE=0.4394 QA=0.2509 KD=2.7754 recon=87.85%
[mem-train] step 200/400 alpha=0.60 loss=1.8751 AE=0.3744 QA=0.3101 KD=1.5264 recon=88.91%
[mem-train] step 300/400 alpha=0.50 loss=2.0138 AE=0.3314 QA=0.2762 KD=1.7100 recon=89.55%
[mem-train] step 400/400 alpha=0.40 loss=2.1436 AE=0.2557 QA=0.1388 KD=1.9580 recon=91.04%
[mem-train] step 1/400 alpha=0.80 loss=7.0673 AE=0.6091 QA=0.6345 KD=6.4531 recon=86.66%
[mem-train] step 100/400 alpha=0.70 loss=2.9213 AE=0.5077 QA=1.4919 KD=2.1191 recon=87.09%
[mem-train] step 200/400 alpha=0.60 loss=2.2579 AE=0.4390 QA=1.3520 KD=1.4541 recon=87.09%
[mem-train] step 300/400 alpha=0.50 loss=1.7689 AE=0.3607 QA=0.8062 KD=1.1855 recon=87.73%
[mem-train] step 400/400 alpha=0.40 loss=1.3069 AE=0.2937 QA=0.3198 KD=0.9976 recon=89.54%
[mem-train] step 1/400 alpha=0.80 loss=14.3121 AE=2.5845 QA=1.8867 KD=11.8672 recon=81.53%
[mem-train] step 100/400 alpha=0.70 loss=4.8847 AE=0.7725 QA=0.4949 KD=4.1953 recon=81.67%
[mem-train] step 200/400 alpha=0.60 loss=3.8783 AE=0.6512 QA=0.0414 KD=3.4707 recon=83.75%
[mem-train] step 300/400 alpha=0.50 loss=2.4524 AE=0.5771 QA=0.0422 KD=2.1426 recon=85.00%
[mem-train] step 400/400 alpha=0.40 loss=1.4645 AE=0.4986 QA=0.1472 KD=1.1768 recon=88.47%
[mem-trunc] full=1406 keep=1024 range=(165,1189) mode=keyword
[mem-train] step 1/400 alpha=0.80 loss=14.8785 AE=0.8128 QA=12.7038 KD=11.6875 recon=75.88%
[mem-train] step 100/400 alpha=0.70 loss=4.5767 AE=0.7448 QA=1.7952 KD=3.5176 recon=77.15%
[mem-train] step 200/400 alpha=0.60 loss=2.4678 AE=0.6515 QA=1.1596 KD=1.6133 recon=79.30%
[mem-train] step 300/400 alpha=0.50 loss=3.3092 AE=0.6459 QA=2.2877 KD=1.8428 recon=80.08%
[mem-train] step 400/400 alpha=0.40 loss=2.4120 AE=0.6259 QA=1.5291 KD=1.2441 recon=80.66%
[mem-train] step 1/400 alpha=0.80 loss=19.2918 AE=1.8752 QA=13.7628 KD=15.0391 recon=60.47%
[mem-train] step 100/400 alpha=0.70 loss=5.4359 AE=1.8922 QA=0.4457 KD=3.9766 recon=58.00%
[mem-train] step 200/400 alpha=0.60 loss=4.6153 AE=1.9095 QA=0.0094 KD=3.4648 recon=57.74%
[mem-train] step 300/400 alpha=0.50 loss=2.6677 AE=1.5940 QA=0.0356 KD=1.8525 recon=61.77%
[mem-train] step 400/400 alpha=0.40 loss=2.1745 AE=1.5247 QA=0.0019 KD=1.5635 recon=63.07%
[10/10] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0  EM/F1(+mem)=0/0.0  recon=63.7%  trainQA=10

=== Final ===
(1) No context     : EM=0.0  F1=0.0
(2) With gold ctx  : EM=40.0  F1=40.0
(3) With mem (AE+QA): EM=0.0  F1=0.0
Saving CSV -> runs/nq_joint_results_kd400step4mem_dif.csv
Saved JSONL -> runs/nq_joint_preds_kd400step4mem_dif.jsonl


### 复杂任务 16mem 10样本
(cram310L20) syt@zoulixin-L20:~/project/cram_L20/DOCAUG$ python -u train_nq_mem_joint_kd.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct   --raw_subset_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/raw/nq_subset_100_dif.jsonl   --aug_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/augmented/nq_multiqa_only_no_unans.jsonl   --device cuda   --max_new_to
kens 16   --temperature 0.0   --mem_len 16   --mem_steps 400   --mem_lr 1e-2   --mem_wd 1e-2   --alpha_start 0.8   --alpha_end 0.4   --mem_l2 1e-4   --mem_noise 0.0   --max_ctx_tokens 1024   --ctx_window keyword   --enable_gc   --enable_kd   --kd_temp 2.0   --kd_lambda 1.0   --kd_first_k 3   --kd_head_weight 2.0   --csv_out runs/nq_joint_results_kd400step16mem_dif.csv   --save_preds runs/nq_joint_preds_kd400step16mem_dif.jsonl   --verbose   --limit 10

(cram310L20) syt@zoulixin-L20:~/project/cram_L20/DOCAUG$ 
(cram310L20) syt@zoulixin-L20:~/project/cram_L20/DOCAUG$ python -u train_nq_mem_joint_kd.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct   --raw_subset_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/raw/nq_subset_100_dif.jsonl   --aug_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/augmented/nq_multiqa_only_no_unans.jsonl   --device cuda   --max_new_to
kens 16   --temperature 0.0   --mem_len 16   --mem_steps 400   --mem_lr 1e-2   --mem_wd 1e-2   --alpha_start 0.8   --alpha_end 0.4   --mem_l2 1e-4   --mem_noise 0.0   --max_ctx_tokens 1024   --ctx_window keyword   --enable_gc   --enable_kd   --kd_temp 2.0   --kd_lambda 1.0   --kd_first_k 3   --kd_head_weight 2.0   --csv_out runs/nq_joint_results_kd400step16mem_dif.csv   --save_preds runs/nq_joint_preds_kd400step16mem_dif.jsonl   --verbose   --limit 10

`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.97s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.94s/it]
[mem-train] step 1/400 alpha=0.80 loss=22.0227 AE=0.9170 QA=6.6016 KD=19.9688 recon=69.36%
[mem-train] step 100/400 alpha=0.70 loss=15.5652 AE=1.2071 QA=1.6465 KD=14.2266 recon=72.11%
[mem-train] step 200/400 alpha=0.60 loss=2.9591 AE=0.9990 QA=0.0044 KD=2.3574 recon=75.00%
[mem-train] step 300/400 alpha=0.50 loss=1.4719 AE=0.7629 QA=0.0009 KD=1.0898 recon=80.20%
[mem-train] step 400/400 alpha=0.40 loss=0.9544 AE=0.7024 QA=0.0001 KD=0.6733 recon=81.07%
[mem-train] step 1/400 alpha=0.80 loss=19.4498 AE=0.9800 QA=4.1101 KD=17.8438 recon=76.45%
[mem-train] step 100/400 alpha=0.70 loss=1.9108 AE=0.7715 QA=0.0003 KD=1.3701 recon=79.19%
[mem-train] step 200/400 alpha=0.60 loss=1.6828 AE=0.7303 QA=0.0002 KD=1.2441 recon=80.78%
[mem-train] step 300/400 alpha=0.50 loss=1.3479 AE=0.6379 QA=0.0008 KD=1.0283 recon=81.94%
[mem-train] step 400/400 alpha=0.40 loss=2.0930 AE=0.5927 QA=0.0008 KD=1.8555 recon=82.23%
[mem-train] step 1/400 alpha=0.80 loss=33.9862 AE=1.9204 QA=5.9994 KD=31.2500 recon=52.44%
[mem-train] step 100/400 alpha=0.70 loss=3.8281 AE=1.1318 QA=0.0123 KD=3.0312 recon=74.03%
[mem-train] step 200/400 alpha=0.60 loss=3.6112 AE=0.9152 QA=0.0321 KD=3.0488 recon=78.08%
[mem-train] step 300/400 alpha=0.50 loss=2.3794 AE=0.7628 QA=0.0053 KD=1.9951 recon=82.14%
[mem-train] step 400/400 alpha=0.40 loss=1.6305 AE=0.7121 QA=0.0016 KD=1.3447 recon=83.44%
[mem-train] step 1/400 alpha=0.80 loss=37.3583 AE=2.7591 QA=6.3797 KD=33.8750 recon=71.82%
[mem-train] step 100/400 alpha=0.70 loss=6.4932 AE=0.6600 QA=0.6247 KD=5.8438 recon=87.12%
[mem-train] step 200/400 alpha=0.60 loss=2.9256 AE=0.6371 QA=0.0001 KD=2.5430 recon=86.12%
[mem-train] step 300/400 alpha=0.50 loss=1.3564 AE=0.6209 QA=0.0019 KD=1.0449 recon=87.27%
[mem-train] step 400/400 alpha=0.40 loss=1.2582 AE=0.4782 QA=0.0203 KD=1.0547 recon=89.13%
[mem-train] step 1/400 alpha=0.80 loss=22.1640 AE=1.2967 QA=4.3050 KD=20.2656 recon=33.62%
[mem-train] step 100/400 alpha=0.70 loss=2.8149 AE=1.8073 QA=0.0020 KD=1.5479 recon=65.17%
[mem-train] step 200/400 alpha=0.60 loss=2.2375 AE=1.5755 QA=0.0011 KD=1.2910 recon=67.76%
[mem-train] step 300/400 alpha=0.50 loss=1.4601 AE=1.2430 QA=0.0007 KD=0.8379 recon=70.52%
[mem-train] step 400/400 alpha=0.40 loss=1.1681 AE=1.0162 QA=0.0014 KD=0.7607 recon=75.52%
[mem-train] step 1/400 alpha=0.80 loss=7.3116 AE=1.0546 QA=0.1913 KD=6.4297 recon=68.44%
[mem-train] step 100/400 alpha=0.70 loss=3.4933 AE=0.4292 QA=0.1931 KD=3.1348 recon=87.21%
[mem-train] step 200/400 alpha=0.60 loss=2.3509 AE=0.3661 QA=1.3133 KD=1.6064 recon=87.85%
[mem-train] step 300/400 alpha=0.50 loss=2.1191 AE=0.2952 QA=0.0229 KD=1.9600 recon=89.87%
[mem-train] step 400/400 alpha=0.40 loss=1.8616 AE=0.2545 QA=0.3662 KD=1.5400 recon=90.83%
[mem-train] step 1/400 alpha=0.80 loss=9.8833 AE=0.7205 QA=1.0661 KD=9.0938 recon=49.31%
[mem-train] step 100/400 alpha=0.70 loss=3.2276 AE=0.5200 QA=0.1955 KD=2.8047 recon=85.91%
[mem-train] step 200/400 alpha=0.60 loss=2.6776 AE=0.4360 QA=0.1802 KD=2.3438 recon=87.51%
[mem-train] step 300/400 alpha=0.50 loss=1.6156 AE=0.3682 QA=0.1070 KD=1.3779 recon=88.58%
[mem-train] step 400/400 alpha=0.40 loss=1.2143 AE=0.3028 QA=0.3629 KD=0.8755 recon=90.39%
[mem-train] step 1/400 alpha=0.80 loss=11.5042 AE=1.9760 QA=1.7655 KD=9.5703 recon=16.67%
[mem-train] step 100/400 alpha=0.70 loss=11.5363 AE=1.1754 QA=2.8773 KD=9.8516 recon=74.17%
[mem-train] step 200/400 alpha=0.60 loss=4.3834 AE=0.7329 QA=0.2541 KD=3.8418 recon=81.81%
[mem-train] step 300/400 alpha=0.50 loss=2.0139 AE=0.5254 QA=0.1370 KD=1.6826 recon=85.83%
[mem-train] step 400/400 alpha=0.40 loss=1.4873 AE=0.3932 QA=0.0699 KD=1.2881 recon=89.58%
[mem-trunc] full=1406 keep=1024 range=(165,1189) mode=keyword
[mem-train] step 1/400 alpha=0.80 loss=17.1313 AE=0.7826 QA=14.5965 KD=13.5859 recon=73.63%
[mem-train] step 100/400 alpha=0.70 loss=2.9423 AE=0.7896 QA=1.3521 KD=1.9844 recon=75.78%
[mem-train] step 200/400 alpha=0.60 loss=2.2854 AE=0.7049 QA=0.6132 KD=1.6172 recon=78.03%
[mem-train] step 300/400 alpha=0.50 loss=2.3977 AE=0.6471 QA=0.1207 KD=2.0137 recon=79.49%
[mem-train] step 400/400 alpha=0.40 loss=1.8823 AE=0.6303 QA=0.9006 KD=1.0898 recon=80.27%
[mem-train] step 1/400 alpha=0.80 loss=19.5271 AE=2.1046 QA=14.0215 KD=15.0391 recon=56.83%
[mem-train] step 100/400 alpha=0.70 loss=5.4616 AE=1.9002 QA=0.2908 KD=4.0430 recon=59.69%
[mem-train] step 200/400 alpha=0.60 loss=3.1197 AE=1.7332 QA=0.0166 KD=2.0723 recon=61.25%
[mem-train] step 300/400 alpha=0.50 loss=2.3239 AE=1.5322 QA=0.0016 KD=1.5566 recon=63.46%
[mem-train] step 400/400 alpha=0.40 loss=1.8828 AE=1.4133 QA=0.0164 KD=1.3076 recon=65.80%
[10/10] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0  EM/F1(+mem)=1/1.0  recon=66.4%  trainQA=10

=== Final ===
(1) No context     : EM=0.0  F1=0.0
(2) With gold ctx  : EM=40.0  F1=40.0
(3) With mem (AE+QA): EM=20.0  F1=20.0
Saving CSV -> runs/nq_joint_results_kd400step16mem_dif.csv
Saved JSONL -> runs/nq_joint_preds_kd400step16mem_dif.jsonl



### 复杂任务 48mem 100样本 tmux：4memdif100
python -u train_nq_mem_joint_kd.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct   --raw_subset_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/raw/nq_subset_100_dif.jsonl   --aug_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/augmented/nq_multiqa_only_no_unans.jsonl   --device cuda   --max_new_tokens 16   --temperature 0.0   --mem_len 48   --mem_steps 400   --mem_lr 1e-2   --mem_wd 1e-2   --alpha_start 0.8   --alpha_end 0.4   --mem_l2 1e-4   --mem_noise 0.0   --max_ctx_tokens 1024   --ctx_window keyword   --enable_gc   --enable_kd   --kd_temp 2.0   --kd_lambda 1.0   --kd_first_k 3   --kd_head_weight 2.0   --csv_out runs/nq_joint_results_kd400step48mem_dif.csv   --save_preds runs/nq_joint_preds_kd400step48mem_dif.jsonl   --verbose 



### 简单任务 4mem 100样本
(cram310L20) syt@zoulixin-L20:~/project/cram_L20/DOCAUG$ python -u train_nq_mem_joint_kd.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct   --raw_subset_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/raw/nq_subset_100.jsonl   --aug_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/augmented/nq_multiqa_only_no_unans_first100.jsonl   --device cuda   --max_new_tokens 16   --temperature 0.0   --mem_len 4   --mem_steps 400   --mem_lr 1e-2   --mem_wd 1e-2   --alpha_start 0.8   --alpha_end 0.4   --mem_l2 1e-4   --mem_noise 0.0   --max_ctx_tokens 1024   --ctx_window keyword   --enable_gc   --enable_kd   --kd_temp 2.0   --kd_lambda 1.0   --kd_first_k 3   --kd_head_weight 2.0   --csv_out runs/nq_joint_results_kd400step_4mem.csv   --save_preds runs/nq_joint_preds_kd400step_4mem.jsonl   --verbose 





### 简单任务 1mem 100样本 无负提升
(cram310L20) syt@zoulixin-L20:~/project/cram_L20/DOCAUG$ python -u train_nq_mem_joint_kd.py   --model /home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct   --raw_subset_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/raw/nq_subset_100.jsonl   --aug_file /home/syt/project/cram_L20/DOCAUG/data/nq_subset/augmented/nq_multiqa_only_no_unans_first100.jsonl   --device cuda   --max_new_tokens 16   --temperature 0.0   --mem_len 1   --mem_steps 200   --mem_lr 1e-2   --mem_wd 1e-2   --alpha_start 0.8   --alpha_end 0.4   --mem_l2 1e-4   --mem_noise 0.0   --max_ctx_tokens 1024   --ctx_window keyword   --enable_gc   --enable_kd   --kd_temp 2.0   --kd_lambda 1.0   --kd_first_k 3   --kd_head_weight 2.0   --csv_out runs/nq_joint_results_kd.csv   --save_preds runs/nq_joint_preds_kd.jsonl   --verbose   --limit 100

`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.88s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.86s/it]
[mem-train] step 1/200 alpha=0.80 loss=14.8749 AE=1.4056 QA=4.6898 KD=12.8125 recon=26.32%
[mem-train] step 100/200 alpha=0.60 loss=3.5837 AE=1.2831 QA=0.7097 KD=2.5293 recon=71.05%
[mem-train] step 200/200 alpha=0.40 loss=2.1060 AE=1.1958 QA=0.4928 KD=1.3320 recon=71.71%
[mem-train] step 1/200 alpha=0.80 loss=31.4553 AE=3.1506 QA=7.1745 KD=27.5000 recon=8.90%
[mem-train] step 100/200 alpha=0.60 loss=14.8107 AE=3.0615 QA=3.8230 KD=11.4453 recon=45.21%
[mem-train] step 200/200 alpha=0.40 loss=5.1594 AE=3.0025 QA=0.3538 KD=3.7461 recon=45.21%
[mem-train] step 1/200 alpha=0.80 loss=16.3068 AE=4.4001 QA=0.8087 KD=12.6250 recon=6.06%
[mem-train] step 100/200 alpha=0.60 loss=8.6524 AE=5.2726 QA=2.9741 KD=4.2969 recon=18.18%
[mem-train] step 200/200 alpha=0.40 loss=5.2969 AE=4.8915 QA=0.4012 KD=3.0996 recon=22.73%
[mem-train] step 1/200 alpha=0.80 loss=30.8496 AE=9.6845 QA=3.7916 KD=22.3438 recon=12.50%
[mem-train] step 100/200 alpha=0.60 loss=3.7324 AE=0.9324 QA=2.3744 KD=2.2246 recon=75.00%
[mem-train] step 200/200 alpha=0.40 loss=1.7234 AE=0.1720 QA=1.0519 KD=1.0234 recon=100.00%
[mem-train] step 1/200 alpha=0.80 loss=9.4550 AE=5.5359 QA=0.2290 KD=4.9805 recon=72.44%
[mem-train] step 100/200 alpha=0.60 loss=2.6997 AE=1.5134 QA=0.0003 KD=1.7900 recon=68.44%
[mem-train] step 200/200 alpha=0.40 loss=1.8172 AE=1.3074 QA=0.0005 KD=1.2939 recon=74.67%
[mem-train] step 1/200 alpha=0.80 loss=22.2928 AE=7.3163 QA=9.6598 KD=14.5078 recon=18.18%
[mem-train] step 100/200 alpha=0.60 loss=1.4524 AE=0.5618 QA=0.0037 KD=1.1133 recon=90.91%
[mem-train] step 200/200 alpha=0.40 loss=0.7757 AE=0.4035 QA=0.0009 KD=0.6138 recon=90.91%
[mem-train] step 1/200 alpha=0.80 loss=8.2709 AE=2.3754 QA=0.5053 KD=6.2695 recon=59.48%
[mem-train] step 100/200 alpha=0.60 loss=1.2805 AE=0.8938 QA=0.0002 KD=0.7432 recon=78.45%
[mem-train] step 200/200 alpha=0.40 loss=0.8458 AE=0.2496 QA=0.0005 KD=0.7456 recon=97.41%
[mem-train] step 1/200 alpha=0.80 loss=29.4956 AE=7.6166 QA=10.9178 KD=21.2188 recon=10.00%
[mem-train] step 100/200 alpha=0.60 loss=3.0579 AE=0.1713 QA=0.7536 KD=2.6543 recon=100.00%
[mem-train] step 200/200 alpha=0.40 loss=1.0712 AE=0.0430 QA=0.0199 KD=1.0420 recon=100.00%
[mem-train] step 1/200 alpha=0.80 loss=22.5759 AE=4.8360 QA=7.1294 KD=17.2812 recon=55.21%
[mem-train] step 100/200 alpha=0.60 loss=3.1351 AE=1.4852 QA=0.0204 KD=2.2344 recon=61.46%
[mem-train] step 200/200 alpha=0.40 loss=1.3789 AE=1.0477 QA=0.0014 KD=0.9590 recon=73.96%
[mem-train] step 1/200 alpha=0.80 loss=34.7543 AE=2.1666 QA=7.2147 KD=31.5781 recon=32.05%
[mem-train] step 100/200 alpha=0.60 loss=16.4145 AE=4.6817 QA=5.9309 KD=11.2344 recon=36.55%
[mem-train] step 200/200 alpha=0.40 loss=9.7494 AE=3.9328 QA=2.1689 KD=6.8750 recon=41.96%
[10/100] EM/F1(no-ctx)=0/0.4  EM/F1(+ctx)=0/0.7  EM/F1(+mem)=0/0.0  recon=56.5%  trainQA=10
[mem-train] step 1/200 alpha=0.80 loss=24.3053 AE=2.9795 QA=5.4679 KD=20.8281 recon=48.72%
[mem-train] step 100/200 alpha=0.60 loss=4.7458 AE=2.1050 QA=0.0446 KD=3.4629 recon=56.41%
[mem-train] step 200/200 alpha=0.40 loss=2.7435 AE=1.3209 QA=0.0721 KD=2.1719 recon=69.23%
[mem-train] step 1/200 alpha=0.80 loss=20.1245 AE=5.9669 QA=2.5361 KD=14.8438 recon=25.70%
[mem-train] step 100/200 alpha=0.60 loss=12.2201 AE=4.4885 QA=2.2689 KD=8.6172 recon=29.32%
[mem-train] step 200/200 alpha=0.40 loss=6.3787 AE=4.2865 QA=0.9962 KD=4.0664 recon=30.12%
[mem-train] step 1/200 alpha=0.80 loss=13.6751 AE=3.7314 QA=1.0669 KD=10.4766 recon=34.00%
[mem-train] step 100/200 alpha=0.60 loss=4.0353 AE=1.5659 QA=0.3877 KD=2.9395 recon=62.00%
[mem-train] step 200/200 alpha=0.40 loss=2.0604 AE=1.3716 QA=0.0977 KD=1.4531 recon=60.00%
[mem-train] step 1/200 alpha=0.80 loss=21.0913 AE=8.2016 QA=2.4549 KD=14.0391 recon=23.62%
[mem-train] step 100/200 alpha=0.60 loss=39.3737 AE=6.6757 QA=4.9790 KD=33.3750 recon=10.24%
[mem-train] step 200/200 alpha=0.40 loss=12.4669 AE=5.8757 QA=2.9027 KD=8.3750 recon=14.96%
[mem-train] step 1/200 alpha=0.80 loss=21.2635 AE=2.9706 QA=4.6696 KD=17.9531 recon=56.93%
[mem-train] step 100/200 alpha=0.60 loss=15.7184 AE=2.5129 QA=1.3049 KD=13.6875 recon=55.47%
[mem-train] step 200/200 alpha=0.40 loss=2.2092 AE=2.2404 QA=0.0009 KD=1.3125 recon=59.85%
[mem-train] step 1/200 alpha=0.80 loss=9.7003 AE=9.3010 QA=0.0670 KD=2.2461 recon=10.24%
[mem-train] step 100/200 alpha=0.60 loss=2.3567 AE=0.7046 QA=0.0016 KD=1.9326 recon=84.88%
[mem-train] step 200/200 alpha=0.40 loss=1.5692 AE=0.3600 QA=0.0007 KD=1.4248 recon=95.12%
[mem-train] step 1/200 alpha=0.80 loss=31.4315 AE=7.1089 QA=5.3623 KD=24.6719 recon=9.41%
[mem-train] step 100/200 alpha=0.60 loss=10.0050 AE=2.3786 QA=0.5416 KD=8.3594 recon=56.47%
[mem-train] step 200/200 alpha=0.40 loss=3.9596 AE=2.1382 QA=0.0372 KD=3.0820 recon=58.82%
[mem-train] step 1/200 alpha=0.80 loss=18.9346 AE=2.8940 QA=4.8547 KD=15.6484 recon=46.45%
[mem-train] step 100/200 alpha=0.60 loss=4.9002 AE=3.1032 QA=0.0097 KD=3.0312 recon=42.18%
[mem-train] step 200/200 alpha=0.40 loss=2.6823 AE=2.6186 QA=0.0034 KD=1.6328 recon=47.87%
[mem-train] step 1/200 alpha=0.80 loss=24.0251 AE=8.6051 QA=13.6741 KD=14.4062 recon=30.99%
[mem-train] step 100/200 alpha=0.60 loss=6.1245 AE=2.3818 QA=0.3663 KD=4.5469 recon=39.44%
[mem-train] step 200/200 alpha=0.40 loss=1.4424 AE=0.5995 QA=0.0024 KD=1.2012 recon=78.87%
[mem-train] step 1/200 alpha=0.80 loss=10.5993 AE=8.0666 QA=0.2807 KD=4.0898 recon=64.10%
[mem-train] step 100/200 alpha=0.60 loss=2.1211 AE=0.4438 QA=0.0023 KD=1.8535 recon=92.31%
[mem-train] step 200/200 alpha=0.40 loss=1.6156 AE=0.2682 QA=0.3084 KD=1.3232 recon=97.44%
[20/100] EM/F1(no-ctx)=0/0.8  EM/F1(+ctx)=1/1.0  EM/F1(+mem)=1/1.0  recon=97.4%  trainQA=9
[mem-train] step 1/200 alpha=0.80 loss=20.5574 AE=6.8469 QA=8.2900 KD=13.4219 recon=9.93%
[mem-train] step 100/200 alpha=0.60 loss=10.5449 AE=3.7708 QA=2.3726 KD=7.3320 recon=32.62%
[mem-train] step 200/200 alpha=0.40 loss=3.4635 AE=3.0212 QA=0.4884 KD=1.9619 recon=43.26%
[mem-train] step 1/200 alpha=0.80 loss=15.9710 AE=8.4931 QA=0.7651 KD=9.0234 recon=5.56%
[mem-train] step 100/200 alpha=0.60 loss=3.3605 AE=2.4016 QA=0.0421 KD=1.9004 recon=55.56%
[mem-train] step 200/200 alpha=0.40 loss=1.1813 AE=0.2287 QA=0.0082 KD=1.0850 recon=100.00%
[mem-train] step 1/200 alpha=0.80 loss=11.6718 AE=7.2226 QA=2.1637 KD=5.4609 recon=3.40%
[mem-train] step 100/200 alpha=0.60 loss=8.2632 AE=1.2315 QA=0.3514 KD=7.3828 recon=71.32%
[mem-train] step 200/200 alpha=0.40 loss=1.5310 AE=1.0734 QA=0.1076 KD=1.0371 recon=73.21%
[mem-train] step 1/200 alpha=0.80 loss=7.0868 AE=3.4363 QA=0.1068 KD=4.3164 recon=28.80%
[mem-train] step 100/200 alpha=0.60 loss=3.8149 AE=1.2342 QA=0.1492 KD=3.0137 recon=71.20%
[mem-train] step 200/200 alpha=0.40 loss=2.7768 AE=0.8190 QA=0.0293 KD=2.4316 recon=84.80%
[mem-train] step 1/200 alpha=0.80 loss=3.9311 AE=3.0668 QA=0.0007 KD=1.4775 recon=77.86%
[mem-train] step 100/200 alpha=0.60 loss=4.0861 AE=1.0714 QA=0.4230 KD=3.2734 recon=76.79%
[mem-train] step 200/200 alpha=0.40 loss=2.7344 AE=0.8952 QA=0.2203 KD=2.2441 recon=81.07%
[mem-train] step 1/200 alpha=0.80 loss=6.3356 AE=5.1737 QA=0.0258 KD=2.1914 recon=61.83%
[mem-train] step 100/200 alpha=0.60 loss=3.4250 AE=1.8084 QA=0.0006 KD=2.3379 recon=51.15%
[mem-train] step 200/200 alpha=0.40 loss=2.6405 AE=1.3001 QA=0.0022 KD=2.1191 recon=70.99%
[mem-train] step 1/200 alpha=0.80 loss=17.7441 AE=4.7158 QA=1.7321 KD=13.6250 recon=16.13%
[mem-train] step 100/200 alpha=0.60 loss=8.8927 AE=5.8547 QA=1.2506 KD=4.8750 recon=22.58%
[mem-train] step 200/200 alpha=0.40 loss=4.9501 AE=5.6563 QA=0.5177 KD=2.3770 recon=23.66%
[mem-train] step 1/200 alpha=0.80 loss=28.5526 AE=7.5127 QA=8.1808 KD=20.9062 recon=6.45%
[mem-train] step 100/200 alpha=0.60 loss=3.7406 AE=0.4348 QA=1.1668 KD=3.0137 recon=93.55%
[mem-train] step 200/200 alpha=0.40 loss=2.2313 AE=0.1860 QA=0.2956 KD=1.9795 recon=96.77%
[mem-train] step 1/200 alpha=0.80 loss=11.1035 AE=1.6732 QA=0.9731 KD=9.5703 recon=68.24%
[mem-train] step 100/200 alpha=0.60 loss=1.9435 AE=0.9599 QA=0.0084 KD=1.3633 recon=80.00%
[mem-train] step 200/200 alpha=0.40 loss=1.0467 AE=0.4911 QA=0.0003 KD=0.8501 recon=90.00%
[mem-train] step 1/200 alpha=0.80 loss=16.2064 AE=1.9224 QA=2.1314 KD=14.2422 recon=19.79%
[mem-train] step 100/200 alpha=0.60 loss=5.9893 AE=4.3491 QA=0.6375 KD=3.1211 recon=26.56%
[mem-train] step 200/200 alpha=0.40 loss=4.0065 AE=3.9785 QA=0.5454 KD=2.0879 recon=32.81%
[30/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0  EM/F1(+mem)=0/0.0  recon=33.3%  trainQA=10
[mem-train] step 1/200 alpha=0.80 loss=8.2418 AE=3.3638 QA=2.7145 KD=5.0078 recon=49.02%
[mem-train] step 100/200 alpha=0.60 loss=6.4029 AE=2.0297 QA=0.4979 KD=4.9844 recon=58.17%
[mem-train] step 200/200 alpha=0.40 loss=3.4679 AE=2.1166 QA=0.0100 KD=2.6152 recon=53.59%
[mem-train] step 1/200 alpha=0.80 loss=42.1013 AE=3.3794 QA=6.8329 KD=38.0312 recon=10.78%
[mem-train] step 100/200 alpha=0.60 loss=21.9261 AE=5.1161 QA=2.7603 KD=17.7500 recon=17.65%
[mem-train] step 200/200 alpha=0.40 loss=8.7553 AE=4.7149 QA=0.0362 KD=6.8477 recon=18.63%
[mem-train] step 1/200 alpha=0.80 loss=44.7722 AE=7.2709 QA=10.0900 KD=36.9375 recon=12.62%
[mem-train] step 100/200 alpha=0.60 loss=14.5055 AE=2.3070 QA=1.1010 KD=12.6797 recon=53.40%
[mem-train] step 200/200 alpha=0.40 loss=1.4425 AE=1.4303 QA=0.0005 KD=0.8701 recon=71.84%
[mem-train] step 1/200 alpha=0.80 loss=22.5579 AE=7.4750 QA=4.2177 KD=15.7344 recon=57.81%
[mem-train] step 100/200 alpha=0.60 loss=4.3103 AE=1.8282 QA=1.1372 KD=2.7578 recon=65.62%
[mem-train] step 200/200 alpha=0.40 loss=2.6926 AE=0.9287 QA=0.9405 KD=1.7568 recon=81.25%
[mem-train] step 1/200 alpha=0.80 loss=8.8127 AE=1.5136 QA=1.5443 KD=7.2930 recon=65.78%
[mem-train] step 100/200 alpha=0.60 loss=3.4982 AE=1.5417 QA=0.0033 KD=2.5703 recon=66.22%
[mem-train] step 200/200 alpha=0.40 loss=2.9284 AE=1.3231 QA=0.0045 KD=2.3965 recon=67.56%
[mem-trunc] full=8211 keep=1024 range=(611,1635) mode=keyword
[mem-train] step 1/200 alpha=0.80 loss=34.4294 AE=2.8105 QA=7.6235 KD=30.6562 recon=44.63%
[mem-train] step 100/200 alpha=0.60 loss=6.9844 AE=2.8493 QA=0.5740 KD=5.0430 recon=44.34%
[mem-train] step 200/200 alpha=0.40 loss=4.9062 AE=2.7799 QA=0.5001 KD=3.4941 recon=43.85%
[mem-train] step 1/200 alpha=0.80 loss=39.9545 AE=7.5665 QA=4.1943 KD=33.0625 recon=45.52%
[mem-train] step 100/200 alpha=0.60 loss=9.3723 AE=2.5162 QA=1.1766 KD=7.3906 recon=47.76%
[mem-train] step 200/200 alpha=0.40 loss=4.0936 AE=2.4048 QA=0.0502 KD=3.1016 recon=49.25%
[mem-train] step 1/200 alpha=0.80 loss=25.1944 AE=2.0882 QA=5.2754 KD=22.4688 recon=13.19%
[mem-train] step 100/200 alpha=0.60 loss=3.4241 AE=1.9700 QA=0.3962 KD=2.0820 recon=56.59%
[mem-train] step 200/200 alpha=0.40 loss=3.8534 AE=1.8225 QA=0.7086 KD=2.6992 recon=53.30%
[mem-train] step 1/200 alpha=0.80 loss=22.3227 AE=3.4893 QA=5.0001 KD=18.5312 recon=72.94%
[mem-train] step 100/200 alpha=0.60 loss=4.4919 AE=1.4256 QA=0.0205 KD=3.6270 recon=71.18%
[mem-train] step 200/200 alpha=0.40 loss=3.0175 AE=1.3056 QA=0.0051 KD=2.4922 recon=76.47%
[mem-train] step 1/200 alpha=0.80 loss=16.5242 AE=1.3318 QA=5.7702 KD=14.3047 recon=77.24%
[mem-train] step 100/200 alpha=0.60 loss=2.2516 AE=0.5192 QA=0.0491 KD=1.9199 recon=90.34%
[mem-train] step 200/200 alpha=0.40 loss=1.5182 AE=0.2617 QA=0.3637 KD=1.1953 recon=94.48%
[40/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0  EM/F1(+mem)=0/0.0  recon=95.9%  trainQA=10
[mem-train] step 1/200 alpha=0.80 loss=37.2441 AE=6.6905 QA=7.8961 KD=30.3125 recon=47.58%
[mem-train] step 100/200 alpha=0.60 loss=8.0365 AE=2.4963 QA=0.4138 KD=6.3711 recon=47.58%
[mem-train] step 200/200 alpha=0.40 loss=2.4682 AE=2.2655 QA=0.0089 KD=1.5566 recon=50.93%
[mem-train] step 1/200 alpha=0.80 loss=27.1510 AE=7.0019 QA=3.5290 KD=20.8438 recon=55.73%
[mem-train] step 100/200 alpha=0.60 loss=7.4080 AE=3.8217 QA=0.1513 KD=5.0508 recon=38.55%
[mem-train] step 200/200 alpha=0.40 loss=2.7806 AE=2.9373 QA=0.0085 KD=1.6006 recon=45.80%
[mem-train] step 1/200 alpha=0.80 loss=10.1409 AE=7.0809 QA=0.3498 KD=4.4062 recon=63.82%
[mem-train] step 100/200 alpha=0.60 loss=2.4749 AE=1.0412 QA=0.0379 KD=1.8340 recon=73.03%
[mem-train] step 200/200 alpha=0.40 loss=2.0036 AE=0.5539 QA=0.0761 KD=1.7363 recon=86.84%
[mem-train] step 1/200 alpha=0.80 loss=18.7903 AE=7.0043 QA=3.7079 KD=12.4453 recon=13.89%
[mem-train] step 100/200 alpha=0.60 loss=6.4588 AE=3.8035 QA=0.6877 KD=3.8984 recon=26.85%
[mem-train] step 200/200 alpha=0.40 loss=4.4871 AE=3.4209 QA=0.3769 KD=2.8926 recon=29.63%
[mem-train] step 1/200 alpha=0.80 loss=10.9357 AE=5.3088 QA=0.0252 KD=6.6836 recon=12.93%
[mem-train] step 100/200 alpha=0.60 loss=4.4791 AE=1.3700 QA=0.3071 KD=3.5332 recon=68.03%
[mem-train] step 200/200 alpha=0.40 loss=3.6365 AE=1.3051 QA=0.3992 KD=2.8750 recon=68.71%
[mem-train] step 1/200 alpha=0.80 loss=6.0511 AE=1.9967 QA=0.2764 KD=4.3984 recon=70.51%
[mem-train] step 100/200 alpha=0.60 loss=3.8036 AE=1.3396 QA=0.0403 KD=2.9824 recon=70.05%
[mem-train] step 200/200 alpha=0.40 loss=2.5378 AE=1.2061 QA=0.0174 KD=2.0449 recon=70.97%
[mem-train] step 1/200 alpha=0.80 loss=4.6086 AE=2.5893 QA=0.0006 KD=2.5371 recon=70.00%
[mem-train] step 100/200 alpha=0.60 loss=2.6549 AE=1.2839 QA=0.0109 KD=1.8789 recon=63.75%
[mem-train] step 200/200 alpha=0.40 loss=1.9592 AE=0.6794 QA=0.3011 KD=1.5068 recon=81.25%
[mem-train] step 1/200 alpha=0.80 loss=9.4365 AE=6.3526 QA=0.4441 KD=4.2656 recon=79.38%
[mem-train] step 100/200 alpha=0.60 loss=2.3938 AE=1.0693 QA=0.5388 KD=1.5361 recon=82.50%
[mem-train] step 200/200 alpha=0.40 loss=1.7410 AE=0.7648 QA=0.3346 KD=1.2344 recon=88.75%
[mem-train] step 1/200 alpha=0.80 loss=31.5735 AE=2.5632 QA=6.1302 KD=28.2969 recon=63.77%
[mem-train] step 100/200 alpha=0.60 loss=2.5897 AE=1.4420 QA=0.0035 KD=1.7217 recon=75.36%
[mem-train] step 200/200 alpha=0.40 loss=1.4907 AE=0.3342 QA=0.0075 KD=1.3525 recon=95.65%
[mem-train] step 1/200 alpha=0.80 loss=19.8059 AE=7.9424 QA=3.4708 KD=12.7578 recon=54.26%
[mem-train] step 100/200 alpha=0.60 loss=9.9528 AE=1.9400 QA=1.4434 KD=8.2109 recon=59.69%
[mem-train] step 200/200 alpha=0.40 loss=3.9208 AE=1.4051 QA=0.5914 KD=3.0039 recon=71.32%
[50/100] EM/F1(no-ctx)=1/1.0  EM/F1(+ctx)=0/0.0  EM/F1(+mem)=1/1.0  recon=71.3%  trainQA=10
[mem-train] step 1/200 alpha=0.80 loss=7.1488 AE=2.7012 QA=0.8764 KD=4.8125 recon=62.50%
[mem-train] step 100/200 alpha=0.60 loss=1.6083 AE=0.4582 QA=0.0364 KD=1.3184 recon=92.50%
[mem-train] step 200/200 alpha=0.40 loss=1.0527 AE=0.1840 QA=0.0189 KD=0.9678 recon=97.50%
[mem-train] step 1/200 alpha=0.80 loss=7.9815 AE=2.3905 QA=4.1932 KD=5.2305 recon=54.39%
[mem-train] step 100/200 alpha=0.60 loss=2.3150 AE=1.9828 QA=0.0006 KD=1.1230 recon=58.77%
[mem-train] step 200/200 alpha=0.40 loss=1.3789 AE=1.4589 QA=0.0022 KD=0.7939 recon=71.93%
[mem-train] step 1/200 alpha=0.80 loss=17.7096 AE=2.6703 QA=1.2654 KD=15.3203 recon=50.00%
[mem-train] step 100/200 alpha=0.60 loss=5.5206 AE=5.3218 QA=0.0048 KD=2.3203 recon=18.06%
[mem-train] step 200/200 alpha=0.40 loss=3.9328 AE=4.6990 QA=0.0041 KD=2.0508 recon=25.00%
[mem-train] step 1/200 alpha=0.80 loss=16.4430 AE=5.0574 QA=2.2589 KD=11.9453 recon=42.95%
[mem-train] step 100/200 alpha=0.60 loss=2.2852 AE=2.5690 QA=0.0001 KD=0.7412 recon=44.49%
[mem-train] step 200/200 alpha=0.40 loss=1.8733 AE=2.4454 QA=0.0076 KD=0.8906 recon=45.81%
[mem-train] step 1/200 alpha=0.80 loss=20.9009 AE=5.9444 QA=1.4693 KD=15.8516 recon=11.84%
[mem-train] step 100/200 alpha=0.60 loss=4.1660 AE=4.9995 QA=0.0052 KD=1.1592 recon=19.18%
[mem-train] step 200/200 alpha=0.40 loss=3.3410 AE=4.7391 QA=0.0017 KD=1.4443 recon=23.27%
[mem-trunc] full=3479 keep=1024 range=(2455,3479) mode=keyword
[mem-train] step 1/200 alpha=0.80 loss=15.7293 AE=1.6067 QA=4.9540 KD=13.4531 recon=69.92%
[mem-train] step 100/200 alpha=0.60 loss=7.4147 AE=1.6102 QA=0.4251 KD=6.2773 recon=69.34%
[mem-train] step 200/200 alpha=0.40 loss=4.1075 AE=1.6153 QA=0.0007 KD=3.4609 recon=69.24%
[mem-train] step 1/200 alpha=0.80 loss=23.7566 AE=6.9036 QA=3.8249 KD=17.4688 recon=53.48%
[mem-train] step 100/200 alpha=0.60 loss=3.7776 AE=2.2666 QA=0.0229 KD=2.4062 recon=54.01%
[mem-train] step 200/200 alpha=0.40 loss=1.7543 AE=1.5764 QA=0.0174 KD=1.1133 recon=68.45%
[mem-train] step 1/200 alpha=0.80 loss=15.7226 AE=1.9673 QA=2.7359 KD=13.6016 recon=63.10%
[mem-train] step 100/200 alpha=0.60 loss=2.5230 AE=0.5067 QA=0.2489 KD=2.1191 recon=90.48%
[mem-train] step 200/200 alpha=0.40 loss=1.9546 AE=0.3080 QA=0.0282 KD=1.8145 recon=96.43%
[mem-train] step 1/200 alpha=0.80 loss=27.5354 AE=7.9062 QA=3.5521 KD=20.5000 recon=2.44%
[mem-train] step 100/200 alpha=0.60 loss=14.9424 AE=4.2053 QA=2.7631 KD=11.3125 recon=24.39%
[mem-train] step 200/200 alpha=0.40 loss=4.5205 AE=2.6213 QA=0.0151 KD=3.4629 recon=51.22%
[mem-train] step 1/200 alpha=0.80 loss=11.3175 AE=7.6821 QA=0.0195 KD=5.1680 recon=6.25%
[mem-train] step 100/200 alpha=0.60 loss=5.0773 AE=4.3674 QA=0.0081 KD=2.4492 recon=25.78%
[mem-train] step 200/200 alpha=0.40 loss=3.3216 AE=2.3410 QA=0.3361 KD=2.1836 recon=57.03%
[60/100] EM/F1(no-ctx)=1/1.0  EM/F1(+ctx)=1/1.0  EM/F1(+mem)=1/1.0  recon=57.0%  trainQA=9
[mem-train] step 1/200 alpha=0.80 loss=35.5327 AE=7.8166 QA=5.8501 KD=28.1094 recon=10.68%
[mem-train] step 100/200 alpha=0.60 loss=7.4254 AE=1.4147 QA=1.6568 KD=5.9141 recon=66.02%
[mem-train] step 200/200 alpha=0.40 loss=1.5889 AE=0.8602 QA=0.0565 KD=1.2109 recon=79.61%
[mem-train] step 1/200 alpha=0.80 loss=8.8050 AE=6.4580 QA=1.4742 KD=3.3438 recon=21.31%
[mem-train] step 100/200 alpha=0.60 loss=12.9948 AE=2.7059 QA=1.0999 KD=10.9297 recon=55.74%
[mem-train] step 200/200 alpha=0.40 loss=2.9583 AE=2.2537 QA=0.2689 KD=1.8955 recon=57.38%
[mem-train] step 1/200 alpha=0.80 loss=14.6177 AE=6.5235 QA=1.1354 KD=9.1719 recon=58.52%
[mem-train] step 100/200 alpha=0.60 loss=2.3655 AE=1.5406 QA=0.1619 KD=1.3750 recon=65.34%
[mem-train] step 200/200 alpha=0.40 loss=1.5038 AE=1.1191 QA=0.1033 KD=0.9941 recon=73.86%
[mem-train] step 1/200 alpha=0.80 loss=7.2341 AE=0.9218 QA=0.9404 KD=6.3086 recon=61.48%
[mem-train] step 100/200 alpha=0.60 loss=2.5741 AE=0.3585 QA=0.0325 KD=2.3457 recon=90.98%
[mem-train] step 200/200 alpha=0.40 loss=1.7072 AE=0.1887 QA=0.1707 KD=1.5293 recon=97.54%
[mem-train] step 1/200 alpha=0.80 loss=20.0502 AE=7.8375 QA=1.9087 KD=13.3984 recon=15.19%
[mem-train] step 100/200 alpha=0.60 loss=3.0943 AE=0.7330 QA=0.0721 KD=2.6250 recon=81.01%
[mem-train] step 200/200 alpha=0.40 loss=1.5959 AE=0.2142 QA=0.0496 KD=1.4805 recon=97.47%
[mem-train] step 1/200 alpha=0.80 loss=22.4479 AE=5.7120 QA=3.1415 KD=17.2500 recon=22.27%
[mem-train] step 100/200 alpha=0.60 loss=4.7942 AE=1.9834 QA=0.8532 KD=3.2617 recon=53.55%
[mem-train] step 200/200 alpha=0.40 loss=2.6316 AE=1.9112 QA=0.0129 KD=1.8594 recon=57.82%
[mem-train] step 1/200 alpha=0.80 loss=25.5280 AE=2.7504 QA=6.0134 KD=22.1250 recon=14.90%
[mem-train] step 100/200 alpha=0.60 loss=3.6970 AE=2.6488 QA=0.0086 KD=2.1016 recon=43.92%
[mem-train] step 200/200 alpha=0.40 loss=2.6731 AE=2.5374 QA=0.0016 KD=1.6572 recon=48.24%
[mem-train] step 1/200 alpha=0.80 loss=26.2416 AE=8.6575 QA=3.3745 KD=18.6406 recon=54.17%
[mem-train] step 100/200 alpha=0.60 loss=3.2202 AE=0.9823 QA=0.0267 KD=2.6191 recon=81.25%
[mem-train] step 200/200 alpha=0.40 loss=1.6400 AE=0.4766 QA=0.0133 KD=1.4414 recon=91.67%
[mem-train] step 1/200 alpha=0.80 loss=15.8969 AE=2.7123 QA=3.1276 KD=13.1016 recon=45.16%
[mem-train] step 100/200 alpha=0.60 loss=7.1277 AE=1.8675 QA=0.4244 KD=5.8359 recon=54.84%
[mem-train] step 200/200 alpha=0.40 loss=5.0174 AE=1.5582 QA=0.1880 KD=4.2812 recon=61.29%
[mem-train] step 1/200 alpha=0.80 loss=4.9226 AE=3.6124 QA=0.0071 KD=2.0312 recon=59.26%
[mem-train] step 100/200 alpha=0.60 loss=2.2417 AE=1.8463 QA=0.0007 KD=1.1318 recon=55.56%
[mem-train] step 200/200 alpha=0.40 loss=3.2264 AE=1.2268 QA=0.1194 KD=2.6641 recon=69.44%
[70/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0  EM/F1(+mem)=0/0.0  recon=70.4%  trainQA=9
[mem-train] step 1/200 alpha=0.80 loss=15.9674 AE=1.8411 QA=5.7145 KD=13.3516 recon=21.11%
[mem-train] step 100/200 alpha=0.60 loss=4.9531 AE=2.0217 QA=0.5967 KD=3.5000 recon=60.00%
[mem-train] step 200/200 alpha=0.40 loss=2.4568 AE=1.8277 QA=0.0344 KD=1.7051 recon=58.89%
[mem-train] step 1/200 alpha=0.80 loss=13.9109 AE=6.7628 QA=0.4720 KD=8.4062 recon=20.41%
[mem-train] step 100/200 alpha=0.60 loss=8.9752 AE=1.1117 QA=1.3180 KD=7.7812 recon=74.49%
[mem-train] step 200/200 alpha=0.40 loss=1.0012 AE=0.6800 QA=0.0004 KD=0.7290 recon=83.67%
[mem-train] step 1/200 alpha=0.80 loss=6.9723 AE=4.7679 QA=0.0573 KD=3.1465 recon=36.00%
[mem-train] step 100/200 alpha=0.60 loss=1.4031 AE=0.3643 QA=0.0159 KD=1.1777 recon=96.00%
[mem-train] step 200/200 alpha=0.40 loss=0.8408 AE=0.1617 QA=0.0110 KD=0.7695 recon=96.00%
[mem-train] step 1/200 alpha=0.80 loss=26.4028 AE=0.6020 QA=5.9341 KD=24.7344 recon=63.27%
[mem-train] step 100/200 alpha=0.60 loss=3.8973 AE=0.9498 QA=0.2161 KD=3.2402 recon=79.62%
[mem-train] step 200/200 alpha=0.40 loss=1.5062 AE=0.8779 QA=0.0142 KD=1.1465 recon=79.82%
[mem-train] step 1/200 alpha=0.80 loss=9.4345 AE=6.7536 QA=0.2268 KD=3.9863 recon=47.83%
[mem-train] step 100/200 alpha=0.60 loss=1.8422 AE=1.1657 QA=0.0515 KD=1.1211 recon=67.39%
[mem-train] step 200/200 alpha=0.40 loss=0.7795 AE=0.2716 QA=0.0015 KD=0.6699 recon=97.83%
[mem-train] step 1/200 alpha=0.80 loss=15.8296 AE=3.3925 QA=2.1404 KD=12.6875 recon=45.45%
[mem-train] step 100/200 alpha=0.60 loss=2.6693 AE=1.9906 QA=0.0056 KD=1.4707 recon=58.18%
[mem-train] step 200/200 alpha=0.40 loss=1.0772 AE=0.2684 QA=0.0010 KD=0.9692 recon=96.36%
[mem-train] step 1/200 alpha=0.80 loss=24.4056 AE=2.4818 QA=2.9603 KD=21.8281 recon=12.09%
[mem-train] step 100/200 alpha=0.60 loss=3.9531 AE=3.7090 QA=0.3655 KD=1.5781 recon=29.67%
[mem-train] step 200/200 alpha=0.40 loss=3.0541 AE=3.0855 QA=0.1718 KD=1.7168 recon=36.26%
[mem-train] step 1/200 alpha=0.80 loss=8.4125 AE=1.0297 QA=2.0841 KD=7.1719 recon=78.53%
[mem-train] step 100/200 alpha=0.60 loss=4.8306 AE=0.8038 QA=0.4403 KD=4.1719 recon=81.94%
[mem-train] step 200/200 alpha=0.40 loss=2.6608 AE=0.7690 QA=0.0775 KD=2.3066 recon=82.72%
[mem-train] step 1/200 alpha=0.80 loss=28.4808 AE=6.5978 QA=5.7003 KD=22.0625 recon=63.22%
[mem-train] step 100/200 alpha=0.60 loss=4.8817 AE=1.1140 QA=0.8890 KD=3.8574 recon=66.67%
[mem-train] step 200/200 alpha=0.40 loss=3.2271 AE=0.7605 QA=0.4868 KD=2.6309 recon=82.76%
[mem-train] step 1/200 alpha=0.80 loss=18.5900 AE=6.7950 QA=2.8012 KD=12.5938 recon=23.60%
[mem-train] step 100/200 alpha=0.60 loss=7.6735 AE=2.1182 QA=0.5142 KD=6.1953 recon=57.30%
[mem-train] step 200/200 alpha=0.40 loss=4.8778 AE=2.0978 QA=0.1589 KD=3.9434 recon=58.43%
[80/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=1/1.0  EM/F1(+mem)=0/0.7  recon=59.6%  trainQA=10
[mem-train] step 1/200 alpha=0.80 loss=5.5565 AE=5.4210 QA=0.0003 KD=1.2197 recon=66.67%
[mem-train] step 100/200 alpha=0.60 loss=2.9951 AE=1.3110 QA=0.0054 KD=2.2051 recon=69.87%
[mem-train] step 200/200 alpha=0.40 loss=1.9234 AE=0.9166 QA=0.0002 KD=1.5566 recon=75.00%
[mem-train] step 1/200 alpha=0.80 loss=30.4390 AE=7.4751 QA=5.4977 KD=23.3594 recon=12.39%
[mem-train] step 100/200 alpha=0.60 loss=6.2828 AE=5.4902 QA=0.2564 KD=2.8809 recon=21.24%
[mem-train] step 200/200 alpha=0.40 loss=5.1044 AE=5.1709 QA=1.2840 KD=2.2656 recon=21.24%
[mem-train] step 1/200 alpha=0.80 loss=20.1538 AE=2.7762 QA=5.4457 KD=16.8438 recon=46.08%
[mem-train] step 100/200 alpha=0.60 loss=3.0763 AE=2.2346 QA=0.0047 KD=1.7314 recon=46.54%
[mem-train] step 200/200 alpha=0.40 loss=2.1500 AE=2.0136 QA=0.0030 KD=1.3428 recon=55.76%
[mem-train] step 1/200 alpha=0.80 loss=14.5432 AE=3.3378 QA=2.4117 KD=11.3906 recon=66.35%
[mem-train] step 100/200 alpha=0.60 loss=2.3278 AE=1.1325 QA=0.1046 KD=1.6055 recon=69.23%
[mem-train] step 200/200 alpha=0.40 loss=1.8312 AE=0.4803 QA=0.6924 KD=1.2236 recon=91.35%
[mem-train] step 1/200 alpha=0.80 loss=10.6872 AE=4.8174 QA=3.0921 KD=6.2148 recon=5.80%
[mem-train] step 100/200 alpha=0.60 loss=4.3922 AE=3.9652 QA=0.0032 KD=2.0078 recon=34.78%
[mem-train] step 200/200 alpha=0.40 loss=1.7773 AE=2.5083 QA=0.0002 KD=0.7739 recon=44.93%
[mem-train] step 1/200 alpha=0.80 loss=8.6457 AE=5.2263 QA=0.0186 KD=4.4609 recon=44.64%
[mem-train] step 100/200 alpha=0.60 loss=4.4541 AE=2.8625 QA=1.2222 KD=2.2461 recon=44.46%
[mem-train] step 200/200 alpha=0.40 loss=3.4079 AE=2.8220 QA=1.3521 KD=1.4678 recon=44.29%
[mem-train] step 1/200 alpha=0.80 loss=3.7086 AE=1.8153 QA=0.0029 KD=2.2559 recon=69.47%
[mem-train] step 100/200 alpha=0.60 loss=2.2132 AE=0.7828 QA=0.6622 KD=1.4785 recon=86.32%
[mem-train] step 200/200 alpha=0.40 loss=1.5820 AE=0.3038 QA=0.3410 KD=1.2559 recon=95.79%
[mem-train] step 1/200 alpha=0.80 loss=28.9796 AE=2.8652 QA=4.9218 KD=25.7031 recon=15.18%
[mem-train] step 100/200 alpha=0.60 loss=8.9307 AE=3.1820 QA=0.7409 KD=6.7227 recon=40.18%
[mem-train] step 200/200 alpha=0.40 loss=5.2216 AE=3.0813 QA=0.5416 KD=3.6641 recon=39.51%
[mem-train] step 1/200 alpha=0.80 loss=13.7452 AE=6.8409 QA=0.4247 KD=8.1875 recon=66.27%
[mem-train] step 100/200 alpha=0.60 loss=3.9381 AE=1.5060 QA=0.6210 KD=2.7852 recon=66.27%
[mem-train] step 200/200 alpha=0.40 loss=1.5562 AE=1.1702 QA=0.0313 KD=1.0693 recon=71.69%
[mem-train] step 1/200 alpha=0.80 loss=22.6285 AE=3.6172 QA=10.5488 KD=17.6250 recon=39.08%
[mem-train] step 100/200 alpha=0.60 loss=5.4928 AE=3.4585 QA=0.1766 KD=3.3438 recon=35.63%
[mem-train] step 200/200 alpha=0.40 loss=2.5890 AE=2.4781 QA=0.0604 KD=1.5615 recon=49.43%
[90/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.0  EM/F1(+mem)=0/0.0  recon=49.4%  trainQA=10
[mem-train] step 1/200 alpha=0.80 loss=11.1270 AE=1.9397 QA=0.6887 KD=9.4375 recon=62.50%
[mem-train] step 100/200 alpha=0.60 loss=2.7145 AE=1.5399 QA=0.0024 KD=1.7881 recon=70.19%
[mem-train] step 200/200 alpha=0.40 loss=1.4277 AE=0.6861 QA=0.0015 KD=1.1523 recon=85.58%
[mem-train] step 1/200 alpha=0.80 loss=8.2984 AE=2.3507 QA=0.6048 KD=6.2969 recon=64.29%
[mem-train] step 100/200 alpha=0.60 loss=4.0383 AE=1.0893 QA=0.1636 KD=3.3184 recon=75.00%
[mem-train] step 200/200 alpha=0.40 loss=2.4087 AE=0.4412 QA=0.1005 KD=2.1719 recon=92.86%
[mem-train] step 1/200 alpha=0.80 loss=21.7999 AE=2.2038 QA=3.4656 KD=19.3438 recon=8.14%
[mem-train] step 100/200 alpha=0.60 loss=3.0600 AE=1.9799 QA=0.0857 KD=1.8359 recon=54.65%
[mem-train] step 200/200 alpha=0.40 loss=2.2429 AE=1.8779 QA=0.2011 KD=1.3711 recon=58.14%
[mem-train] step 1/200 alpha=0.80 loss=17.5933 AE=2.2407 QA=2.8705 KD=15.2266 recon=54.89%
[mem-train] step 100/200 alpha=0.60 loss=3.3394 AE=1.9460 QA=0.0830 KD=2.1367 recon=52.63%
[mem-train] step 200/200 alpha=0.40 loss=2.1470 AE=1.7122 QA=0.0834 KD=1.4121 recon=61.65%
[mem-train] step 1/200 alpha=0.80 loss=37.6981 AE=3.9663 QA=9.5001 KD=32.6250 recon=74.16%
[mem-train] step 100/200 alpha=0.60 loss=3.9902 AE=1.8009 QA=0.2781 KD=2.7969 recon=66.67%
[mem-train] step 200/200 alpha=0.40 loss=2.0597 AE=1.5945 QA=0.0032 KD=1.4199 recon=69.51%
[mem-train] step 1/200 alpha=0.80 loss=13.0927 AE=7.5196 QA=3.5101 KD=6.3750 recon=11.88%
[mem-train] step 100/200 alpha=0.60 loss=5.1609 AE=3.7553 QA=0.0335 KD=2.8906 recon=28.71%
[mem-train] step 200/200 alpha=0.40 loss=3.9702 AE=3.5479 QA=0.0037 KD=2.5488 recon=32.67%
[mem-train] step 1/200 alpha=0.80 loss=46.6180 AE=1.1337 QA=9.8050 KD=43.7500 recon=34.19%
[mem-train] step 100/200 alpha=0.60 loss=23.6604 AE=6.1766 QA=5.5486 KD=17.7344 recon=18.71%
[mem-train] step 200/200 alpha=0.40 loss=12.3927 AE=4.9714 QA=2.5225 KD=8.8906 recon=23.87%
[mem-train] step 1/200 alpha=0.80 loss=20.7927 AE=6.5684 QA=3.7834 KD=14.7812 recon=54.29%
[mem-train] step 100/200 alpha=0.60 loss=2.8737 AE=1.5823 QA=0.0611 KD=1.8984 recon=61.14%
[mem-train] step 200/200 alpha=0.40 loss=2.0869 AE=1.2138 QA=0.0224 KD=1.5879 recon=69.71%
[mem-train] step 1/200 alpha=0.80 loss=38.9405 AE=7.3740 QA=13.2533 KD=30.3906 recon=15.87%
[mem-train] step 100/200 alpha=0.60 loss=9.7798 AE=5.3282 QA=1.7312 KD=5.8867 recon=17.46%
[mem-train] step 200/200 alpha=0.40 loss=4.8686 AE=4.7199 QA=0.2021 KD=2.8594 recon=21.43%
[mem-train] step 1/200 alpha=0.80 loss=20.2815 AE=6.0767 QA=4.4051 KD=14.5391 recon=50.00%
[mem-train] step 100/200 alpha=0.60 loss=4.9981 AE=2.0488 QA=0.3651 KD=3.6211 recon=50.00%
[mem-train] step 200/200 alpha=0.40 loss=3.3738 AE=1.1619 QA=0.5581 KD=2.5742 recon=75.56%
[100/100] EM/F1(no-ctx)=0/0.0  EM/F1(+ctx)=0/0.5  EM/F1(+mem)=0/0.0  recon=76.7%  trainQA=10

=== Final ===
(1) No context     : EM=15.0  F1=27.3
(2) With gold ctx  : EM=44.0  F1=57.7
(3) With mem (AE+QA): EM=18.0  F1=30.3
Saving CSV -> runs/nq_joint_results_kd.csv
Saved JSONL -> runs/nq_joint_preds_kd.jsonl
