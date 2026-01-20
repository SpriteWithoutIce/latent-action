#!/bin/bash
set -e

BASE_DIR=/home/linyihan/linyh/datasets/xarm_tfds_out
SCRIPT=/home/linyihan/linyh/latent-action/xarm.py   # 你的python脚本路径（按实际改）

GPU0=2
GPU1=3

TASK=xarm_tabletop
DATA_DIR="$BASE_DIR/$TASK/1.0.0"

echo "==============================="
echo "TASK: $TASK"
echo "DATA_DIR: $DATA_DIR"
echo "GPU0=$GPU0 GPU1=$GPU1"
echo "==============================="

# 收集所有 train 分片
FILES=($(ls "$DATA_DIR"/"$TASK"-train.tfrecord-* 2>/dev/null || true))

N=${#FILES[@]}
if [ $N -eq 0 ]; then
  echo "[ERROR] No tfrecord shards found in $DATA_DIR"
  exit 1
fi

echo "[INFO] Found $N shards"

# 按奇偶分配到两个 GPU 并行跑
for ((i=0; i<$N; i++)); do
  FILE=${FILES[$i]}
  if (( i % 2 == 0 )); then
    echo "[GPU $GPU0] launch $FILE"
    CUDA_VISIBLE_DEVICES=$GPU0 python "$SCRIPT" "$FILE" &
  else
    echo "[GPU $GPU1] launch $FILE"
    CUDA_VISIBLE_DEVICES=$GPU1 python "$SCRIPT" "$FILE" &
  fi

  # 控制并发：最多同时跑2个进程
  if (( (i+1) % 2 == 0 )); then
    wait
  fi
done

# 收尾：如果总数是奇数，会剩一个没 wait
wait

echo "✅ ALL DONE"
