from modelscope import snapshot_download
import os

model_id = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
# LLM-Research/Meta-Llama-3.1-8B-Instruct
cache_dir = os.path.expanduser("/home/syt/project/Cram/model/model_scope_model")
os.makedirs(cache_dir, exist_ok=True)

model_dir = snapshot_download(model_id, cache_dir=cache_dir)  # 1.10 版本不支持 local_dir
print("downloaded to:", model_dir)

# 可选：建一个软链接到你项目目录，方便后续加载
target = "/home/syt/project/Cram/model/model_scope_model/llama3_1_8b_instruct"
if not os.path.exists(target):
    os.symlink(model_dir, target, target_is_directory=True)
    print("linked:", target, "->", model_dir)
