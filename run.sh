#!/bin/bash
# ==========================================
# 自动处理所有分片 TFRecord 文件
# ==========================================

DATA_DIR=/home/linyihan/linyh/tensorflow_datasets/move_playingcard_away/1.0.0
export CUDA_VISIBLE_DEVICES=0

for FILE in $DATA_DIR/lift_pot-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

for FILE in $DATA_DIR/lift_pot-val.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done