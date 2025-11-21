import tensorflow as tf
import sys

path = sys.argv[1]

ds = tf.data.TFRecordDataset(path)
raw = next(iter(ds))

example = tf.train.Example()
example.ParseFromString(raw.numpy())

print("=== Keys in TFRecord Example ===")
for k in example.features.feature.keys():
    print(k)
