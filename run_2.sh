#!/bin/bash
# ==========================================
# 自动处理所有分片 TFRecord 文件
# ==========================================

DATA_DIR=/home/linyihan/linyh/datasets/RoboTwin/click_bell/1.0.0
export CUDA_VISIBLE_DEVICES=3

for FILE in $DATA_DIR/click_bell-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

for FILE in $DATA_DIR/click_bell-val.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

DATA_DIR=/home/linyihan/linyh/datasets/RoboTwin/dump_bin_bigbin/1.0.0

for FILE in $DATA_DIR/dump_bin_bigbin-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

for FILE in $DATA_DIR/dump_bin_bigbin-val.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

DATA_DIR=/home/linyihan/linyh/datasets/RoboTwin/grab_roller/1.0.0

for FILE in $DATA_DIR/grab_roller-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

for FILE in $DATA_DIR/grab_roller-val.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

DATA_DIR=/home/linyihan/linyh/datasets/RoboTwin/open_laptop/1.0.0

for FILE in $DATA_DIR/open_laptop-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

for FILE in $DATA_DIR/open_laptop-val.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

DATA_DIR=/home/linyihan/linyh/datasets/RoboTwin/open_laptop/1.0.0

for FILE in $DATA_DIR/open_laptop-train.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done

for FILE in $DATA_DIR/open_laptop-val.tfrecord-*; do
  echo "🚀 开始处理: $FILE"
  python augment_with_latent.py "$FILE"
  echo "✅ 完成: $FILE"
  echo "-----------------------------"
done