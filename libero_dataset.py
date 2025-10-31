"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).
"""

import os
import torch
import draccus
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass
from pathlib import Path

from data_provider.action_tokenizer import ActionTokenizer
from data_provider.data_utils import RLDSBatchTransformInternVL, PaddedCollatorForActionPrediction
from data_provider.datasets import RLDSDataset

from latent_action_model.genie.modules import ControllableDINOLatentActionModel
from data_provider.utils import set_global_seed
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_lam(cfg, device_id=0) -> ControllableDINOLatentActionModel:
    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=768,
        latent_dim=128,
        num_latents=16,
        patch_size=14,
        enc_blocks=12,
        dec_blocks=12,
        num_heads=12,
        dropout=0.,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()
    
    return latent_action_model

@dataclass
class FinetuneConfig:
    seed: int = 42
    # vlm
    vlm_path: str = "/home/linyihan/linyh/VLM/InternVL3_5-1B"
    lam_path: str = "/home/linyihan/linyh/latent-action/latent_action_model/checkpoints/lam-stage-2.ckpt"

    # data
    data_root_dir: str = "datasets/open-x-embodiment"
    data_mix: str = "libero_goal"
    window_size: int = 8

    # training
    max_steps: int = 10
    per_device_batch_size: int = 4

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    # latent action model
    latent_action_model = load_lam(cfg)

    # Create Action Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_path, trust_remote_code=True, use_fast=False)
    action_tokenizer = ActionTokenizer(tokenizer)

    batch_transform = RLDSBatchTransformInternVL(
        action_tokenizer,
        tokenizer,
        use_wrist_image=True,
        use_proprio=True,
    )
    
    vla_dataset = RLDSDataset(
        Path(cfg.data_root_dir),
        cfg.data_mix,
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=100_000,
        train=True,
        image_aug=True,
        window_size=cfg.window_size,
        goal_image_step=cfg.window_size,
    )
    # Create Collator
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, 
        tokenizer.pad_token_id, 
        padding_side="right"
    )
    print('Dataset length = ', len(vla_dataset))
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
    total_steps = cfg.max_steps
    # Train!
    with tqdm.tqdm(total=total_steps, leave=False) as progress:
        for batch_idx, batch in enumerate(dataloader):
            batch = batch
            print(batch["input_ids"].shape)


if __name__ == "__main__":
    finetune()
