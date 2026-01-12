#!/bin/bash
# ==========================================
# 自动处理所有分片 TFRecord 文件
# ==========================================

#!/bin/bash

BASE_DIR=/home/linyihan/linyh/datasets/RoboTwin
export CUDA_VISIBLE_DEVICES=3

# 手动指定要处理的任务
declare -A DATA_DIRS=(
  ["move_playingcard_away"]="move_playingcard_away"
  ["pick_dual_bottles"]="pick_dual_bottles"
  ["pick_diverse_bottles"]="pick_diverse_bottles"
  ["place_container_plate"]="place_container_plate"
  ["place_a2b_left"]="place_a2b_left"
)

for TASK in "${!DATA_DIRS[@]}"; do
  DATA_DIR="$BASE_DIR/${DATA_DIRS[$TASK]}/1.0.0"

  echo "==============================="
  echo "📂 处理任务: $TASK"
  echo "📁 数据目录: $DATA_DIR"
  echo "==============================="

  # echo "$DATA_DIR"/"$TASK"-train.tfrecord-*
  # train
  for FILE in "$DATA_DIR"/"$TASK"-train.tfrecord-*; do
    [ -e "$FILE" ] || continue
    echo "🚀 开始处理 (train): $FILE"
    python augment_with_latent.py "$FILE"
    echo "✅ 完成: $FILE"
    echo "-----------------------------"
  done

  # val
  for FILE in "$DATA_DIR"/"$TASK"-val.tfrecord-*; do
    [ -e "$FILE" ] || continue
    echo "🚀 开始处理 (val): $FILE"
    python augment_with_latent.py "$FILE"
    echo "✅ 完成: $FILE"
    echo "-----------------------------"
  done
done
