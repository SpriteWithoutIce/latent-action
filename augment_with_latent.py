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
LATENT_IDX_FEATURE = {
    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
    "tensor": {
        "shape": {
            "dimensions": ["4"]
        },
        "dtype": "float32",
        "encoding": "none"
    },
    "description": "Latent action indices from tokenizer."
}

LATENT_Z_FEATURE = {
    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
    "tensor": {
        "shape": {
            "dimensions": ["4", "128"]
        },
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

###############################################
# 1. Load latent model
###############################################

LAMBDA_CKPT = "/home/linyihan/linyh/latent-action/latent_action_model/logs/robotwin_0108_lam_stage2/last.ckpt"

def load_lam():
    model = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=768, latent_dim=128,
        num_latents=16, patch_size=14,
        enc_blocks=12, dec_blocks=12,
        num_heads=12, dropout=0.0,
    )
    ckpt = torch.load(LAMBDA_CKPT, map_location="cpu")["state_dict"]
    ckpt = {k.replace("lam.", ""): v for k, v in ckpt.items()}
    missing_keys, unexpected_keys = model.load_state_dict(
        ckpt,
        strict=False
    )

    if len(missing_keys) > 0:
        raise RuntimeError(
            f"❌ Missing keys when loading checkpoint:\n" +
            "\n".join(missing_keys)
        )

    if len(unexpected_keys) > 0:
        print(
            f"⚠️ Ignored unexpected keys ({len(unexpected_keys)}):\n" +
            "\n".join(unexpected_keys)
        )

    return model.eval()

lam = load_lam().to("cuda:0")
to_tensor = transforms.ToTensor()


###############################################
# 2. Flatten TFRecord schema
###############################################

FEATURES = {
    "steps/is_first": tf.io.VarLenFeature(tf.int64),
    "steps/is_last": tf.io.VarLenFeature(tf.int64),
    "steps/is_terminal": tf.io.VarLenFeature(tf.int64),

    "steps/action": tf.io.VarLenFeature(tf.float32),
    "steps/discount": tf.io.VarLenFeature(tf.float32),
    "steps/reward": tf.io.VarLenFeature(tf.float32),

    "steps/observation/state": tf.io.VarLenFeature(tf.float32),

    "steps/observation/image": tf.io.VarLenFeature(tf.string),
    "steps/observation/left_wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/observation/right_wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/observation/low_cam_image": tf.io.VarLenFeature(tf.string),

    # RaggedTensor flatten form:
    "steps/language_instruction/ragged_flat_values": tf.io.VarLenFeature(tf.string),
    "steps/language_instruction/ragged_row_lengths_0": tf.io.VarLenFeature(tf.int64),

    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
}


###############################################
# 3. Decode single episode from flattened format
###############################################
def parse_flatten_episode(example):

    # convert sparse to dense
    dense = {}
    for k, v in example.items():
        if isinstance(v, tf.SparseTensor):
            dense[k] = tf.sparse.to_dense(v)
        else:
            dense[k] = v

    # length
    T = dense["steps/is_first"].shape[0]

    # language_instruction (ragged)
    flat_vals = dense["steps/language_instruction/ragged_flat_values"].numpy()
    row_len = dense["steps/language_instruction/ragged_row_lengths_0"].numpy()

    # reconstruct ragged indices
    lang = []
    idx = 0
    for t in range(T):
        L = row_len[t]
        lang.append([flat_vals[idx + i].decode() for i in range(L)])
        idx += L

    # parse per-step dict
    traj = []
    for t in range(T):
        traj.append({
            "is_first": int(dense["steps/is_first"][t]),
            "is_last": int(dense["steps/is_last"][t]),
            "is_terminal": int(dense["steps/is_terminal"][t]),

            "action": dense["steps/action"].numpy().reshape(T, 14)[t],
            "discount": float(dense["steps/discount"][t]),
            "reward": float(dense["steps/reward"][t]),

            "state": dense["steps/observation/state"].numpy().reshape(T,14)[t],

            "image": dense["steps/observation/image"][t].numpy(),
            "left_wrist_image": dense["steps/observation/left_wrist_image"][t].numpy(),
            "right_wrist_image": dense["steps/observation/right_wrist_image"][t].numpy(),
            "low_cam_image": dense["steps/observation/low_cam_image"][t].numpy(),

            "language_instruction": lang[t],
        })

    return traj, dense["episode_metadata/file_path"].numpy()


###############################################
# 4. latent encoding
###############################################
# def encode_latents(traj):
#     lat_idx = []
#     lat_z = []

#     for t in range(len(traj)):
#         img = Image.open(io.BytesIO(traj[t]["image"])).convert("RGB").resize((224,224))
#         tn = min(t+11, len(traj)-1)
#         img2 = Image.open(io.BytesIO(traj[tn]["image"])).convert("RGB").resize((224,224))

#         video = torch.stack([to_tensor(img), to_tensor(img2)], 0).unsqueeze(0).to("cuda:0")

#         with torch.no_grad():
#             out = lam.vq_encode(video)
#             lat_idx.append(out["indices"].squeeze().cpu().numpy().astype(np.float32))   # (4,)
#             lat_z.append(out["z"].squeeze().cpu().numpy().astype(np.float32))           # (4,128)

#     return np.array(lat_idx), np.array(lat_z)

def encode_latents(traj, batch_size=32):
    """
    Safe batch inference version.
    - 不会 OOM
    - 不会一次把 T 全部丢进 GPU
    - 性能仍然比逐帧快 5~20 倍
    """

    # Pre-decode images and construct pairs
    frames = []
    for t in range(len(traj)):
        img = Image.open(io.BytesIO(traj[t]["image"])).convert("RGB").resize((224,224))
        tn = min(t+11, len(traj)-1)
        img2 = Image.open(io.BytesIO(traj[tn]["image"])).convert("RGB").resize((224,224))
        frames.append(torch.stack([to_tensor(img), to_tensor(img2)], dim=0))

    frames = torch.stack(frames, dim=0)      # (T,2,3,224,224)

    lat_idx_list = []
    lat_z_list   = []

    # ----- Batch inference -----
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size].to("cuda:0")   # (B,2,3,224,224)

            out = lam.vq_encode(batch)

            lat_idx_list.append(out["indices"].cpu())
            lat_z_list.append(out["z"].cpu())

    # Concatenate results
    lat_idx = torch.cat(lat_idx_list, dim=0).numpy().astype(np.float32)
    lat_z   = torch.cat(lat_z_list, dim=0).numpy().astype(np.float32)

    return lat_idx, lat_z


###############################################
# 5. write flattened TFRecord with added latent
###############################################
def write_episode_flat(writer, traj, file_path, lat_idx, lat_z):

    T = len(traj)

    def FL(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def IL(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def BL(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    # Language ragged flatten
    flat_vals = []
    row_lengths = []
    for t in range(T):
        row_lengths.append(len(traj[t]["language_instruction"]))
        for s in traj[t]["language_instruction"]:
            flat_vals.append(s.encode())

    feature = {
        "steps/is_first": IL([int(s["is_first"]) for s in traj]),
        "steps/is_last": IL([int(s["is_last"]) for s in traj]),
        "steps/is_terminal": IL([int(s["is_terminal"]) for s in traj]),

        "steps/action": FL(np.array([s["action"] for s in traj]).flatten().tolist()),
        "steps/discount": FL([float(s["discount"]) for s in traj]),
        "steps/reward": FL([float(s["reward"]) for s in traj]),

        "steps/observation/state": FL(np.array([s["state"] for s in traj]).flatten().tolist()),

        "steps/observation/image": BL([s["image"] for s in traj]),
        "steps/observation/left_wrist_image": BL([s["left_wrist_image"] for s in traj]),
        "steps/observation/right_wrist_image": BL([s["right_wrist_image"] for s in traj]),
        "steps/observation/low_cam_image": BL([s["low_cam_image"] for s in traj]),

        # Ragged
        "steps/language_instruction/ragged_flat_values": BL(flat_vals),
        "steps/language_instruction/ragged_row_lengths_0": IL(row_lengths),

        # Added latent fields
        "steps/latent_idx": FL(lat_idx.flatten().tolist()),
        "steps/latent_z": FL(lat_z.flatten().tolist()),

        "episode_metadata/file_path": BL([file_path]),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


###############################################
# 6. batch process multiple TFRecord files
###############################################
if len(sys.argv) < 2:
    print("usage: python batch_augment_flattened_tfrecord.py '/path/to/*.tfrecord'")
    sys.exit(1)

input_arg = sys.argv[1]
if os.path.isdir(input_arg):
    files = sorted(glob.glob(os.path.join(input_arg, "*.tfrecord")))
else:
    files = sorted(glob.glob(input_arg))

if not files:
    print("No TFRecord files found.")
    sys.exit(1)

output_dir = os.path.join(os.path.dirname(files[0]), "output")
os.makedirs(output_dir, exist_ok=True)

BASE_OUTPUT_DIR = "/home/linyihan/linyh/datasets/robotwin_latent_0108"

for i, fpath in enumerate(files):
    print(f"\n===== Processing {i+1}/{len(files)}: {fpath} =====")

    dataset = tf.data.TFRecordDataset(fpath)
    # out_path = os.path.join(output_dir, os.path.basename(fpath))
    # writer = tf.io.TFRecordWriter(out_path)
    
    parts = fpath.split(os.sep)

    # 取任务名和版本号
    task_name = parts[-3]     # 2_bowls
    version   = parts[-2]     # 1.0.0

    output_dir = os.path.join(BASE_OUTPUT_DIR, task_name, version)
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== 复制 dataset_info.json =====
    src_root = os.path.dirname(fpath)
    src_dataset_info = os.path.join(src_root, "dataset_info.json")
    dst_dataset_info = os.path.join(output_dir, "dataset_info.json")

    if os.path.exists(src_dataset_info) and not os.path.exists(dst_dataset_info):
        shutil.copy2(src_dataset_info, dst_dataset_info)

    # ===== 复制 + patch feature.json =====
    src_feature_json = os.path.join(src_root, "features.json")
    dst_feature_json = os.path.join(output_dir, "features.json")

    if os.path.exists(src_feature_json):
        copy_and_patch_feature_json(src_feature_json, dst_feature_json)

    out_path = os.path.join(output_dir, os.path.basename(fpath))
    writer = tf.io.TFRecordWriter(out_path)

    for raw in tqdm(dataset, desc="episodes"):
        example = tf.io.parse_single_example(raw, FEATURES)
        traj, file_path = parse_flatten_episode(example)
        lat_idx, lat_z = encode_latents(traj)
        write_episode_flat(writer, traj, file_path, lat_idx, lat_z)

    writer.close()

print("\nALL DONE.")
