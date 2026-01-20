import tensorflow as tf
import numpy as np
import os, sys, glob, io
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from latent_action_model.genie.modules import ControllableDINOLatentActionModel

import json
import shutil
from collections import OrderedDict


# =====================================================
# 0) feature patch: add latent fields to features.json
# =====================================================
LATENT_IDX_FEATURE = {
    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
    "tensor": {
        "shape": {"dimensions": ["4"]},
        "dtype": "float32",
        "encoding": "none"
    },
    "description": "Latent action indices from tokenizer."
}

LATENT_Z_FEATURE = {
    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
    "tensor": {
        "shape": {"dimensions": ["4", "128"]},
        "dtype": "float32",
        "encoding": "none"
    },
    "description": "Continuous latent action embedding."
}


def copy_and_patch_feature_json(src_feature_json, dst_feature_json):
    with open(src_feature_json, "r") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    steps_features = (
        data["featuresDict"]["features"]
            ["steps"]["sequence"]["feature"]
            ["featuresDict"]["features"]
    )

    # 已经加过就直接写出（支持断点重跑）
    if "latent_idx" in steps_features:
        with open(dst_feature_json, "w") as f:
            json.dump(data, f, indent=2)
        return

    new_steps = OrderedDict()
    for k, v in steps_features.items():
        new_steps[k] = v
        if k == "reward":
            new_steps["latent_idx"] = LATENT_IDX_FEATURE
            new_steps["latent_z"] = LATENT_Z_FEATURE

    (
        data["featuresDict"]["features"]
            ["steps"]["sequence"]["feature"]
            ["featuresDict"]["features"]
    ) = new_steps

    with open(dst_feature_json, "w") as f:
        json.dump(data, f, indent=2)


# =====================================================
# 1) Load latent model
# =====================================================
LAMBDA_CKPT = "/home/linyihan/linyh/latent-action/latent_action_model/logs/xarm_0116/last.ckpt"
# ⬆️ 你现在训练了新的 LAM，就把这个路径改成你新 ckpt 的路径！

def load_lam():
    model = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=768, latent_dim=128,
        num_latents=16, patch_size=14,
        enc_blocks=12, dec_blocks=12,
        num_heads=12, dropout=0.0,
    )
    ckpt = torch.load(LAMBDA_CKPT, map_location="cpu")["state_dict"]
    ckpt = {k.replace("lam.", ""): v for k, v in ckpt.items()}

    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)

    if len(missing_keys) > 0:
        raise RuntimeError("❌ Missing keys:\n" + "\n".join(missing_keys))

    if len(unexpected_keys) > 0:
        print(f"⚠️ Ignored unexpected keys ({len(unexpected_keys)}):\n" + "\n".join(unexpected_keys))

    return model.eval()


lam = load_lam().to("cuda:0")
to_tensor = transforms.ToTensor()


# =====================================================
# 2) Flatten TFRecord schema (XARM)
# =====================================================
# ✅ 这里严格按你 xarm 的 feature 来写
FEATURES = {
    "steps/is_first": tf.io.VarLenFeature(tf.int64),
    "steps/is_last": tf.io.VarLenFeature(tf.int64),
    "steps/is_terminal": tf.io.VarLenFeature(tf.int64),

    "steps/action": tf.io.VarLenFeature(tf.float32),
    "steps/discount": tf.io.VarLenFeature(tf.float32),
    "steps/reward": tf.io.VarLenFeature(tf.float32),

    "steps/observation/state": tf.io.VarLenFeature(tf.float32),

    # 图像：tfds Image -> tf.string (jpeg bytes)
    "steps/observation/wrist": tf.io.VarLenFeature(tf.string),
    "steps/observation/primary": tf.io.VarLenFeature(tf.string),

    # Text：一般是每 step 一个 string
    "steps/language_instruction": tf.io.VarLenFeature(tf.string),

    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
}


# =====================================================
# 3) Decode single episode from flattened format
# =====================================================
def parse_flatten_episode(example, state_dim=7, action_dim=7):
    # sparse -> dense
    dense = {}
    for k, v in example.items():
        if isinstance(v, tf.SparseTensor):
            dense[k] = tf.sparse.to_dense(v)
        else:
            dense[k] = v

    T = dense["steps/is_first"].shape[0]

    # (T,7)
    action = dense["steps/action"].numpy().reshape(T, action_dim)
    state = dense["steps/observation/state"].numpy().reshape(T, state_dim)

    wrist = dense["steps/observation/wrist"].numpy()     # (T,) bytes
    primary = dense["steps/observation/primary"].numpy() # (T,) bytes

    # language: 每步一个 string
    lang_raw = dense["steps/language_instruction"].numpy()
    lang = []
    for t in range(T):
        if t < len(lang_raw):
            v = lang_raw[t]
            if isinstance(v, (bytes, np.bytes_)):
                lang.append(v.decode("utf-8"))
            else:
                lang.append(str(v))
        else:
            lang.append("")

    traj = []
    for t in range(T):
        traj.append({
            "is_first": int(dense["steps/is_first"][t]),
            "is_last": int(dense["steps/is_last"][t]),
            "is_terminal": int(dense["steps/is_terminal"][t]),

            "action": action[t].astype(np.float32),
            "discount": float(dense["steps/discount"][t]),
            "reward": float(dense["steps/reward"][t]),

            "state": state[t].astype(np.float32),

            "wrist": wrist[t],
            "primary": primary[t],

            "language_instruction": lang[t],
        })

    return traj, dense["episode_metadata/file_path"].numpy()


# =====================================================
# 4) latent encoding
# =====================================================
def encode_latents(traj, image_key="primary", batch_size=32, jump=11):
    """
    和 robotwin 完全一致：用 (t, t+11) 两帧作为输入，算 latent action
    image_key:
      - "primary" 推荐（front view）
      - "wrist" 也可以试
    """
    frames = []
    for t in range(len(traj)):
        img1 = Image.open(io.BytesIO(traj[t][image_key])).convert("RGB").resize((224, 224))
        tn = min(t + jump, len(traj) - 1)
        img2 = Image.open(io.BytesIO(traj[tn][image_key])).convert("RGB").resize((224, 224))
        frames.append(torch.stack([to_tensor(img1), to_tensor(img2)], dim=0))

    frames = torch.stack(frames, dim=0)  # (T,2,3,224,224)

    lat_idx_list = []
    lat_z_list = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size].to("cuda:0")
            out = lam.vq_encode(batch)
            lat_idx_list.append(out["indices"].cpu())
            lat_z_list.append(out["z"].cpu())

    lat_idx = torch.cat(lat_idx_list, dim=0).numpy().astype(np.float32)  # (T,4)
    lat_z = torch.cat(lat_z_list, dim=0).numpy().astype(np.float32)      # (T,4,128)
    return lat_idx, lat_z


# =====================================================
# 5) write flattened TFRecord with added latent
# =====================================================
def write_episode_flat(writer, traj, file_path, lat_idx, lat_z, state_dim=7, action_dim=7):
    T = len(traj)

    def FL(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def IL(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def BL(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    feature = {
        "steps/is_first": IL([int(s["is_first"]) for s in traj]),
        "steps/is_last": IL([int(s["is_last"]) for s in traj]),
        "steps/is_terminal": IL([int(s["is_terminal"]) for s in traj]),

        "steps/action": FL(np.array([s["action"] for s in traj]).reshape(T, action_dim).flatten().tolist()),
        "steps/discount": FL([float(s["discount"]) for s in traj]),
        "steps/reward": FL([float(s["reward"]) for s in traj]),

        "steps/observation/state": FL(np.array([s["state"] for s in traj]).reshape(T, state_dim).flatten().tolist()),

        "steps/observation/wrist": BL([s["wrist"] for s in traj]),
        "steps/observation/primary": BL([s["primary"] for s in traj]),

        "steps/language_instruction": BL([s["language_instruction"].encode("utf-8") for s in traj]),

        # ✅ 新增 latent
        "steps/latent_idx": FL(lat_idx.flatten().tolist()),  # (T,4)
        "steps/latent_z": FL(lat_z.flatten().tolist()),      # (T,4,128)

        "episode_metadata/file_path": BL([file_path]),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


# =====================================================
# 6) batch process tfrecord
# =====================================================
if len(sys.argv) < 2:
    print("usage: python batch_augment_xarm_latent_tfrecord.py '/path/to/*.tfrecord*'")
    sys.exit(1)

input_arg = sys.argv[1]
if os.path.isdir(input_arg):
    files = sorted(glob.glob(os.path.join(input_arg, "*.tfrecord*")))
else:
    files = sorted(glob.glob(input_arg))

if not files:
    print("No TFRecord files found.")
    sys.exit(1)

BASE_OUTPUT_DIR = "/home/linyihan/linyh/datasets/xarm_latent_labeled"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

STATE_DIM = 7
ACTION_DIM = 7

for i, fpath in enumerate(files):
    print(f"\n===== Processing {i+1}/{len(files)}: {fpath} =====")

    dataset = tf.data.TFRecordDataset(fpath)

    # 输入一般是：
    # .../xarm_tfds_out/xarm_tabletop/1.0.0/xarm_tabletop-train.tfrecord-00000-of-00001
    parts = fpath.split(os.sep)
    dataset_name = parts[-3]  # xarm_tabletop
    version = parts[-2]       # 1.0.0

    output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name, version)
    os.makedirs(output_dir, exist_ok=True)

    # ===== copy dataset_info.json =====
    src_root = os.path.dirname(fpath)
    src_dataset_info = os.path.join(src_root, "dataset_info.json")
    dst_dataset_info = os.path.join(output_dir, "dataset_info.json")
    if os.path.exists(src_dataset_info) and not os.path.exists(dst_dataset_info):
        shutil.copy2(src_dataset_info, dst_dataset_info)

    # ===== copy + patch features.json =====
    src_feature_json = os.path.join(src_root, "features.json")
    dst_feature_json = os.path.join(output_dir, "features.json")
    if os.path.exists(src_feature_json):
        copy_and_patch_feature_json(src_feature_json, dst_feature_json)

    out_path = os.path.join(output_dir, os.path.basename(fpath))
    writer = tf.io.TFRecordWriter(out_path)

    for raw in tqdm(dataset, desc="episodes"):
        example = tf.io.parse_single_example(raw, FEATURES)
        traj, file_path = parse_flatten_episode(example, state_dim=STATE_DIM, action_dim=ACTION_DIM)

        lat_idx, lat_z = encode_latents(
            traj,
            image_key="primary",   # ✅ 推荐 primary
            batch_size=32,
            jump=11,
        )

        write_episode_flat(
            writer,
            traj,
            file_path,
            lat_idx,
            lat_z,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
        )

    writer.close()

print("\nALL DONE.")
