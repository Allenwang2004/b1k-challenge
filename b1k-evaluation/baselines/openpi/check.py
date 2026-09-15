import jax
from openpi.shared import download # 或使用 openpi 的 checkpoint 載入工具
from orbax.checkpoint import PyTreeCheckpointer

# 指向你的 Checkpoint 路徑（通常含有 default/ 或 manifest 等檔名）
ckpt_path = "/home/b1k-challenge/evaluation/behavior_checkpoints/30ep"

checkpointer = PyTreeCheckpointer()
# 唯讀模式載入原始權重樹結構
raw_params = checkpointer.restore(ckpt_path)

print("=== Checkpoint 頂層 Keys ===")
print(raw_params.keys())