#!/bin/bash
# ==========================================
# 自动处理所有分片 TFRecord 文件
# ==========================================

DATA_DIR=/home/linyihan/linyh/datasets/modified_libero_rlds/libero_10_no_noops/1.0.0
export CUDA_VISIBLE_DEVICES=1

for FILE in $DATA_DIR/liber_o10-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done
