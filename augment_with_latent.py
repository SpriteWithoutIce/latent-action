import tensorflow as tf
import numpy as np
import os, sys, glob, io
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from latent_action_model.genie.modules import ControllableDINOLatentActionModel


###############################################
# 1. Load latent model
###############################################

LAMBDA_CKPT = "/home/linyihan/linyh/latent-action/latent_action_model/checkpoints/lam-stage-2.ckpt"

def load_lam():
    model = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=768, latent_dim=128,
        num_latents=16, patch_size=14,
        enc_blocks=12, dec_blocks=12,
        num_heads=12, dropout=0.0,
    )
    ckpt = torch.load(LAMBDA_CKPT, map_location="cpu")["state_dict"]
    ckpt = {k.replace("lam.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
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
def encode_latents(traj):
    lat_idx = []
    lat_z = []

    for t in range(len(traj)):
        img = Image.open(io.BytesIO(traj[t]["image"])).convert("RGB").resize((224,224))
        tn = min(t+11, len(traj)-1)
        img2 = Image.open(io.BytesIO(traj[tn]["image"])).convert("RGB").resize((224,224))

        video = torch.stack([to_tensor(img), to_tensor(img2)], 0).unsqueeze(0).to("cuda:0")

        with torch.no_grad():
            out = lam.vq_encode(video)
            lat_idx.append(out["indices"].squeeze().cpu().numpy().astype(np.float32))   # (4,)
            lat_z.append(out["z"].squeeze().cpu().numpy().astype(np.float32))           # (4,128)

    return np.array(lat_idx), np.array(lat_z)


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


for i, fpath in enumerate(files):
    print(f"\n===== Processing {i+1}/{len(files)}: {fpath} =====")

    dataset = tf.data.TFRecordDataset(fpath)
    out_path = os.path.join(output_dir, os.path.basename(fpath))
    writer = tf.io.TFRecordWriter(out_path)

    for raw in tqdm(dataset, desc="episodes"):
        example = tf.io.parse_single_example(raw, FEATURES)
        traj, file_path = parse_flatten_episode(example)
        lat_idx, lat_z = encode_latents(traj)
        write_episode_flat(writer, traj, file_path, lat_idx, lat_z)

    writer.close()

print("\nALL DONE.")
