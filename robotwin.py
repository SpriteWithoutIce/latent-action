import tensorflow as tf
import os, sys, io
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from latent_action_model.genie.modules import ControllableDINOLatentActionModel

# ============================================
# 1️⃣ 设置文件路径
# ============================================
if len(sys.argv) < 2:
    print("❌ 请提供 TFRecord 文件路径，例如：python latent.py /path/to/file.tfrecord")
    sys.exit(1)

tfrecord_path = sys.argv[1]
print(f"读取文件: {tfrecord_path}")

# ============================================
# 2️⃣ 读取 TFRecord
# ============================================
dataset = tf.data.TFRecordDataset(tfrecord_path)

# ============================================
# 3️⃣ 定义字段结构
# ============================================
feature_description = {
    "steps/is_first": tf.io.VarLenFeature(tf.int64),
    "steps/is_last": tf.io.VarLenFeature(tf.int64),
    "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
    "steps/action": tf.io.VarLenFeature(tf.float32),
    "steps/discount": tf.io.VarLenFeature(tf.float32),
    "steps/reward": tf.io.VarLenFeature(tf.float32),
    "steps/observation/state": tf.io.VarLenFeature(tf.float32),
    # "steps/observation/joint_state": tf.io.VarLenFeature(tf.float32),
    "steps/observation/image": tf.io.VarLenFeature(tf.string),
    # "steps/observation/wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/observation/left_wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/observation/right_wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/observation/low_cam_image": tf.io.VarLenFeature(tf.string),
    
    "steps/language_instruction": tf.io.VarLenFeature(tf.string),
    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
}

def parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    for k, v in example.items():
        if isinstance(v, tf.SparseTensor):
            example[k] = tf.sparse.to_dense(v)
    return example


# ============================================
# 4️⃣ 解析轨迹
# ============================================
trajectories = []
for i, raw_record in enumerate(dataset):
    ex = parse_example(raw_record)
    num_steps = len(ex["steps/is_first"])

    # ---- 按 features.json 定义的维度 reshape ----
    action = ex["steps/action"].numpy().astype(np.float32).reshape(num_steps, 14)
    state = ex["steps/observation/state"].numpy().astype(np.float32).reshape(num_steps, 14)

    steps = []
    for t in range(num_steps):
        step_data = {
            "is_first": int(ex["steps/is_first"][t]),
            "is_last": int(ex["steps/is_last"][t]),
            "is_terminal": int(ex["steps/is_terminal"][t]),
            "action": action[t],           # ✅ (7,)
            "reward": float(ex["steps/reward"][t]),
            "discount": float(ex["steps/discount"][t]),
            "state": state[t],             # ✅ (8,)
            "language_instruction": (
                ex["steps/language_instruction"].numpy()[0].decode("utf-8")
                if len(ex["steps/language_instruction"]) > 0 else ""
            ),
            "image": ex["steps/observation/image"].numpy()[t] if len(ex["steps/observation/image"]) > t else b"",
            "left_wrist_image": ex["steps/observation/left_wrist_image"].numpy()[t] if len(ex["steps/observation/left_wrist_image"]) > t else b"",
            "right_wrist_image": ex["steps/observation/right_wrist_image"].numpy()[t] if len(ex["steps/observation/right_wrist_image"]) > t else b"",
            "low_cam_image": ex["steps/observation/low_cam_image"].numpy()[t] if len(ex["steps/observation/low_cam_image"]) > t else b"",
        }
        steps.append(step_data)

    trajectories.append(steps)
    print(f"✅ 读取轨迹 {i}，共 {num_steps} 步")

print(f"\n📊 文件中共 {len(trajectories)} 条轨迹")


# ============================================
# 5️⃣ 加载 latent 模型
# ============================================
lam_path = "/home/linyihan/linyh/latent-action/latent_action_model/checkpoints/lam-stage-2.ckpt"

def load_lam() -> ControllableDINOLatentActionModel:
    model = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=768, latent_dim=128,
        num_latents=16, patch_size=14, enc_blocks=12,
        dec_blocks=12, num_heads=12, dropout=0.0,
    )
    ckpt = torch.load(lam_path, map_location="cpu")["state_dict"]
    ckpt = {k.replace("lam.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return model.eval()

latent_action_model = load_lam().to("cuda:0")
tokenizer = latent_action_model
to_tensor = transforms.ToTensor()


# ============================================
# 6️⃣ 生成 latent idx + z
# ============================================
all_latent_indices, all_latent_z = [], []

for traj_idx, traj in enumerate(tqdm(trajectories, desc="Processing trajectories")):
    traj_latent_idx, traj_latent_z = [], []
    num_steps = len(traj)

    for t in range(num_steps):
        img_bytes = traj[t]["image"]
        if not img_bytes:
            continue
        t_next = min(t + 11, num_steps - 1)
        img_next_bytes = traj[t_next]["image"]

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224), Image.BICUBIC)
        img_k = Image.open(io.BytesIO(img_next_bytes)).convert("RGB").resize((224, 224), Image.BICUBIC)

        video = torch.stack([to_tensor(img), to_tensor(img_k)], dim=0).unsqueeze(0).to("cuda:0")

        with torch.no_grad():
            vq_output = tokenizer.vq_encode(video)
            latent_idx = vq_output["indices"].squeeze().cpu().numpy().astype(np.float32)  # (4,)
            latent_z = vq_output["z"].squeeze().cpu().numpy().astype(np.float32)          # (4,128)

        traj_latent_idx.append(latent_idx)
        traj_latent_z.append(latent_z)

    all_latent_indices.append(traj_latent_idx)
    all_latent_z.append(traj_latent_z)
    print(f"✅ 轨迹 {traj_idx} 完成，共 {len(traj_latent_idx)} 步")

print(f"\n🎯 共处理 {len(all_latent_indices)} 条轨迹")


# ============================================
# 7️⃣ 写回 TFRecord
# ============================================
output_dir = os.path.join(os.path.dirname(tfrecord_path), "output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, os.path.basename(tfrecord_path))
print(f"💾 开始写入新文件: {output_path}")

def _bytes_feature(value):
    if isinstance(value, (list, np.ndarray)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.tobytes() if isinstance(v, np.ndarray) else v for v in value]))
    elif isinstance(value, bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value).encode()]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

with tf.io.TFRecordWriter(output_path) as writer:
    for traj_idx, traj in enumerate(trajectories):
        num_steps = len(traj)
        actions = np.stack([s["action"] for s in traj], axis=0).astype(np.float32)  # (steps,7)
        states = np.stack([s["state"] for s in traj], axis=0).astype(np.float32)    # (steps,8)

        latents = np.stack(all_latent_indices[traj_idx], axis=0).astype(np.float32)  # (steps,4)
        z = np.stack(all_latent_z[traj_idx], axis=0).astype(np.float32)              # (steps,4,128)

        feature = {
            "steps/is_first": _int64_feature([int(s["is_first"]) for s in traj]),
            "steps/is_last": _int64_feature([int(s["is_last"]) for s in traj]),
            "steps/is_terminal": _int64_feature([int(s["is_terminal"]) for s in traj]),
            "steps/action": _float_feature(actions.flatten().tolist()),
            "steps/discount": _float_feature([float(s["discount"]) for s in traj]),
            "steps/reward": _float_feature([float(s["reward"]) for s in traj]),
            "steps/observation/state": _float_feature(states.flatten().tolist()),
            "steps/language_instruction": _bytes_feature([s["language_instruction"].encode() for s in traj]),
            "steps/observation/image": _bytes_feature([s["image"] for s in traj]),
            "steps/observation/left_wrist_image": _bytes_feature([s["left_wrist_image"] for s in traj]),
            "steps/observation/right_wrist_image": _bytes_feature([s["right_wrist_image"] for s in traj]),
            "steps/observation/low_cam_image": _bytes_feature([s["low_cam_image"] for s in traj]),
            "episode_metadata/file_path": _bytes_feature([traj[0]["language_instruction"].encode() if traj[0]["language_instruction"] else b""]),
            "steps/latent_idx": _float_feature(latents.flatten().tolist()),
            "steps/latent_z": _float_feature(z.flatten().tolist()),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        print(f"✅ 写入轨迹 {traj_idx} | steps={num_steps}, latent_z={z.shape}")

print(f"\n🎉 新文件保存完成: {output_path}")



# import tensorflow as tf
# import os, sys, io
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import torch
# from torchvision import transforms
# from latent_action_model.genie.modules import ControllableDINOLatentActionModel

# if len(sys.argv) < 2:
#     print("❌ 用法：python process_aloha.py '/path/to/*.tfrecord'")
#     sys.exit(1)

# tfrecord_pattern = sys.argv[1]
# input_files = tf.io.gfile.glob(tfrecord_pattern)

# output_dir = os.path.join(os.path.dirname(input_files[0]), "output")
# os.makedirs(output_dir, exist_ok=True)

# print("📁 输出目录:", output_dir)

# # ========== latent model same as before ==========
# lam_path = "/home/linyihan/linyh/latent-action/latent_action_model/checkpoints/lam-stage-2.ckpt"

# def load_lam():
#     model = ControllableDINOLatentActionModel(
#         in_dim=3, model_dim=768, latent_dim=128,
#         num_latents=16, patch_size=14, enc_blocks=12,
#         dec_blocks=12, num_heads=12, dropout=0.0,
#     )
#     ckpt = torch.load(lam_path, map_location="cpu")["state_dict"]
#     ckpt = {k.replace("lam.", ""): v for k, v in ckpt.items()}
#     model.load_state_dict(ckpt)
#     return model.eval()

# lam = load_lam().to("cuda:0")
# to_tensor = transforms.ToTensor()

# # ========== TFDS per-step feature ==========

# step_feature_description = {
#     "observation/image": tf.io.FixedLenFeature([], tf.string),
#     "observation/left_wrist_image": tf.io.FixedLenFeature([], tf.string),
#     "observation/right_wrist_image": tf.io.FixedLenFeature([], tf.string),
#     "observation/low_cam_image": tf.io.FixedLenFeature([], tf.string),

#     "observation/state": tf.io.FixedLenFeature([14], tf.float32),
#     "action": tf.io.FixedLenFeature([14], tf.float32),
#     "discount": tf.io.FixedLenFeature([], tf.float32),
#     "reward": tf.io.FixedLenFeature([], tf.float32),

#     "is_first": tf.io.FixedLenFeature([], tf.int64),
#     "is_last": tf.io.FixedLenFeature([], tf.int64),
#     "is_terminal": tf.io.FixedLenFeature([], tf.int64),

#     "language_instruction": tf.io.VarLenFeature(tf.string),
# }

# def parse_episode(example_proto):
#     episode = tf.io.parse_single_example(
#         example_proto,
#         {
#             "steps": tf.io.VarLenFeature(tf.string),
#             "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
#         },
#     )
#     steps_bytes = episode["steps"].values
#     steps = []
#     for raw in steps_bytes:
#         step = tf.io.parse_single_example(raw, step_feature_description)
#         steps.append(step)
#     return steps

# # =====================================================
# # 🚀 MAIN LOOP: process each file separately
# # =====================================================

# for file_idx, file_path in enumerate(input_files):
#     print(f"\n=============================")
#     print(f"📄 处理文件 {file_idx}: {file_path}")
#     print("=============================\n")

#     dataset = tf.data.TFRecordDataset(file_path)
#     episodes = []

#     # ---------- read episodes ----------
#     for record in dataset:
#         eps = []
#         for s in parse_episode(record):
#             eps.append({
#                 "is_first": int(s["is_first"].numpy()),
#                 "is_last": int(s["is_last"].numpy()),
#                 "is_terminal": int(s["is_terminal"].numpy()),

#                 "action": s["action"].numpy(),
#                 "reward": float(s["reward"].numpy()),
#                 "discount": float(s["discount"].numpy()),

#                 "state": s["observation/state"].numpy(),

#                 "image": s["observation/image"].numpy(),
#                 "left_wrist_image": s["observation/left_wrist_image"].numpy(),
#                 "right_wrist_image": s["observation/right_wrist_image"].numpy(),
#                 "low_cam_image": s["observation/low_cam_image"].numpy(),

#                 "language_instruction": [
#                     t.decode() for t in s["language_instruction"].values.numpy()
#                 ],
#             })
#         episodes.append(eps)

#     print(f"📊 本文件 episodes 数量：{len(episodes)}")

#     # ---------- latent encoding ----------
#     all_idx = []
#     all_z = []

#     for traj in tqdm(episodes, desc="latent encoding"):
#         traj_idx, traj_z = [], []

#         for t in range(len(traj)):
#             img = Image.open(io.BytesIO(traj[t]["image"])).convert("RGB").resize((224, 224))
#             tn = min(t + 11, len(traj)-1)
#             img_next = Image.open(io.BytesIO(traj[tn]["image"])).convert("RGB").resize((224, 224))

#             video = torch.stack([to_tensor(img), to_tensor(img_next)], 0).unsqueeze(0).to("cuda:0")

#             with torch.no_grad():
#                 out = lam.vq_encode(video)
#                 traj_idx.append(out["indices"].squeeze().cpu().numpy())
#                 traj_z.append(out["z"].squeeze().cpu().numpy())

#         all_idx.append(traj_idx)
#         all_z.append(traj_z)

#     # ---------- write output ----------
#     out_file = os.path.join(output_dir, os.path.basename(file_path))
#     print("💾 写入新文件：", out_file)

#     writer = tf.io.TFRecordWriter(out_file)

#     for traj, idxs, zs in zip(episodes, all_idx, all_z):
#         step_serialized = []

#         for s, li, lz in zip(traj, idxs, zs):

#             feature = {
#                 "observation/image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s["image"]])),
#                 "observation/left_wrist_image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s["left_wrist_image"]])),
#                 "observation/right_wrist_image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s["right_wrist_image"]])),
#                 "observation/low_cam_image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[s["low_cam_image"]])),
#                 "observation/state": tf.train.Feature(float_list=tf.train.FloatList(value=s["state"])),

#                 "action": tf.train.Feature(float_list=tf.train.FloatList(value=s["action"])),
#                 "reward": tf.train.Feature(float_list=tf.train.FloatList(value=[s["reward"]])),
#                 "discount": tf.train.Feature(float_list=tf.train.FloatList(value=[s["discount"]])),
#                 "is_first": tf.train.Feature(int64_list=tf.train.Int64List(value=[s["is_first"]])),
#                 "is_last": tf.train.Feature(int64_list=tf.train.Int64List(value=[s["is_last"]])),
#                 "is_terminal": tf.train.Feature(int64_list=tf.train.Int64List(value=[s["is_terminal"]])),
#                 "language_instruction": tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.encode() for t in s["language_instruction"]])),

#                 "latent_idx": tf.train.Feature(float_list=tf.train.FloatList(value=li.flatten())),
#                 "latent_z": tf.train.Feature(float_list=tf.train.FloatList(value=lz.flatten())),
#             }

#             step_serialized.append(
#                 tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
#             )

#         episode_feature = {
#             "steps": tf.train.Feature(bytes_list=tf.train.BytesList(value=step_serialized)),
#             "episode_metadata/file_path": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"aloha"])),
#         }

#         writer.write(tf.train.Example(features=tf.train.Features(feature=episode_feature)).SerializeToString())

#     writer.close()
#     print("🎉 完成！")

# print("\n🌟 全部文件处理完成！")
