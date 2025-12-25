#!/bin/bash
# ==========================================
# 自动处理所有分片 TFRecord 文件
# ==========================================

DATA_DIR=/home/linyihan/linyh/datasets/jaka/wipe/1.0.0
export CUDA_VISIBLE_DEVICES=2

for FILE in $DATA_DIR/wipe-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python jaka.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

DATA_DIR=/home/linyihan/linyh/datasets/jaka/2_bowls/1.0.0

for FILE in $DATA_DIR/2_bowls-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python jaka.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

DATA_DIR=/home/linyihan/linyh/datasets/jaka/3_bowls/1.0.0

for FILE in $DATA_DIR/3_bowls-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python jaka.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

DATA_DIR=/home/linyihan/linyh/datasets/jaka/4_bowls/1.0.0

for FILE in $DATA_DIR/4_bowls-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python jaka.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

# for FILE in $DATA_DIR/place_a2b_left-val.tfrecord-*; do
#   echo "🚀 开始处理: $FILE"
#   python augment_with_latent.py "$FILE"
#   echo "✅ 完成: $FILE"
#   echo "-----------------------------"
# done