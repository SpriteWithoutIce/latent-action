"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).
"""

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import draccus
import torch
import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from data_provider.action_tokenizer import ActionTokenizer
from data_provider.data_utils import RLDSBatchTransformInternVL, PaddedCollatorForActionPrediction
from data_provider.datasets import RLDSDataset

from latent_action_model.genie.modules import ControllableDINOLatentActionModel
from data_provider.utils import set_global_seed
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DATA_ROOT_MAP = {
    "libero_spatial": "/home/linyihan/linyh/datasets/modified_libero_rlds/libero_spatial_no_noops/1.0.0",
    "libero_object": "/home/linyihan/linyh/datasets/modified_libero_rlds/libero_object_no_noops/1.0.0",
    "libero_goal": "/home/linyihan/linyh/datasets/modified_libero_rlds/libero_goal_no_noops/1.0.0",
    "libero_10": "/home/linyihan/linyh/datasets/modified_libero_rlds/libero_10_no_noops/1.0.0",
}

def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Inverse ImageNet normalization for a single image tensor."""

    if image.dim() == 4:
        mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(image.device)
        std = IMAGENET_STD.view(1, 3, 1, 1).to(image.device)
    elif image.dim() == 3:
        mean = IMAGENET_MEAN.to(image.device)
        std = IMAGENET_STD.to(image.device)
    else:
        raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")

    return (image * std + mean).clamp(0, 1)


class LatentActionGalleryCollector:
    """Collects dataset samples grouped by latent action and exports image grids."""

    def __init__(
        self,
        output_dir: Path,
        max_images_per_latent: int,
        patch_index: int,
        max_total_images: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir
        self.max_images_per_latent = max_images_per_latent
        self.patch_index = patch_index
        self.max_total_images = max_total_images
        self.images_by_latent: DefaultDict[Tuple[int, ...], List[torch.Tensor]] = defaultdict(list)
        self.sample_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        self.total_saved = 0

    def add_batch(self, batch: Dict[str, torch.Tensor]) -> int:
        if self.max_total_images is not None and self.total_saved >= self.max_total_images:
            return 0

        pixel_values = batch["pixel_values"]
        if pixel_values.dim() != 5:
            raise ValueError(
                "Expected pixel_values to have shape [B, num_patches, C, H, W], "
                f"got {tuple(pixel_values.shape)}"
            )

        if self.patch_index >= pixel_values.shape[1]:
            raise ValueError(
                f"Patch index {self.patch_index} is out of bounds for tensor with "
                f"{pixel_values.shape[1]} patches"
            )

        latent_actions: Sequence[Sequence[int]] = self._to_latent_sequences(batch["latent_action_idx"])

        new_images = 0
        for sample_idx, latent_sequence in enumerate(latent_actions):
            latent_key = tuple(int(v) for v in latent_sequence)
            if len(self.images_by_latent[latent_key]) >= self.max_images_per_latent:
                continue

            if self.max_total_images is not None and self.total_saved >= self.max_total_images:
                break

            patch = pixel_values[sample_idx, self.patch_index].to(torch.float32)
            patch = denormalize_image(patch)
            self.images_by_latent[latent_key].append(patch.cpu())
            self.sample_counts[latent_key] += 1
            self.total_saved += 1
            new_images += 1

        return new_images

    def _to_latent_sequences(self, latent_action_idx: Sequence) -> List[Sequence[int]]:
        sequences: List[Sequence[int]] = []
        for latent in latent_action_idx:
            if torch.is_tensor(latent):
                sequences.append(latent.detach().cpu().tolist())
            else:
                sequences.append(list(latent))
        return sequences

    def is_saturated(self) -> bool:
        return self.max_total_images is not None and self.total_saved >= self.max_total_images

    def save(self) -> None:
        if not self.images_by_latent:
            print("No latent action images collected; nothing to save.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = []

        for latent_key, images in sorted(self.images_by_latent.items()):
            if not images:
                continue
            if len(images) < 10:
                continue
            grid = make_grid(
                torch.stack(images, dim=0),
                nrow=self._grid_nrow(len(images)),
                padding=2,
            ).permute(1, 2, 0)

            grid_np = grid.numpy()
            output_path = self.output_dir / self._filename_for_latent(latent_key)
            plt.imsave(output_path, grid_np)

            metadata.append(
                {
                    "latent_action": list(latent_key),
                    "saved_images": len(images),
                    "total_samples_seen": self.sample_counts[latent_key],
                    "filename": output_path.name,
                }
            )

        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        print(f"Saved {len(metadata)} latent action galleries to {self.output_dir}.")

    @staticmethod
    def _grid_nrow(num_images: int) -> int:
        return max(1, int(math.ceil(math.sqrt(num_images))))

    @staticmethod
    def _filename_for_latent(latent_key: Tuple[int, ...]) -> str:
        latent_str = "-".join(f"{value:02d}" for value in latent_key)
        return f"latent_{latent_str}.png"

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
    data_mix: List[str] = field(default_factory=list)
    window_size: int = 8
    goal_image_step: int = 32

    # training
    max_steps: int = 10
    per_device_batch_size: int = 4
    # visualization
    visualize_latent_actions: bool = True
    latent_action_output_dir: str = "latent_action_visualizations"
    max_images_per_latent: int = 16
    visualization_patch_index: int = 0
    visualization_max_batches: Optional[int] = None
    visualization_max_total_images: Optional[int] = None

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
        latent_action_model,
        use_wrist_image=True,
        use_proprio=True,
    )
    

    datasets = []

    # # 支持传入字符串或列表
    # if isinstance(cfg.data_mix, str):
    #     dataset_names = [cfg.data_mix]
    # else:
    #     dataset_names = cfg.data_mix

    # for mix_name in dataset_names:
    #     print(f"Loading dataset: {mix_name}")
    #     ds = RLDSDataset(
    #         Path(DATA_ROOT_MAP[mix_name]),
    #         mix_name,
    #         batch_transform,
    #         resize_resolution=(224, 224),
    #         shuffle_buffer_size=100_000,
    #         train=True,
    #         image_aug=True,
    #         window_size=cfg.window_size,
    #         goal_image_step=cfg.window_size,
    #     )
    #     datasets.append(ds)

    # # ✅ 合并所有子数据集
    # vla_dataset = ConcatDataset(datasets)
    # print(f"Total combined dataset length = {len(vla_dataset)}")
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
    print(f"Total combined dataset length = {len(vla_dataset)}")
    # Create Collator
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, 
        tokenizer.pad_token_id, 
        padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
    total_steps = len(vla_dataset) // cfg.per_device_batch_size
    print("total_steps: ", total_steps)
    if cfg.visualize_latent_actions:
        collector = LatentActionGalleryCollector(
            output_dir=Path(cfg.latent_action_output_dir),
            max_images_per_latent=cfg.max_images_per_latent,
            patch_index=cfg.visualization_patch_index,
            max_total_images=cfg.visualization_max_total_images,
        )

        progress_total = (
            cfg.visualization_max_batches
            if cfg.visualization_max_batches is not None
            else total_steps
        )

        with tqdm.tqdm(total=progress_total, leave=False) as progress:
            for batch_idx, batch in enumerate(dataloader):
                collector.add_batch(batch)
                progress.update(1)
                progress.set_postfix(
                    images=collector.total_saved,
                    latents=len(collector.images_by_latent),
                )

                if collector.is_saturated():
                    break

                if (
                    cfg.visualization_max_batches is not None
                    and batch_idx + 1 >= cfg.visualization_max_batches
                ):
                    break

        collector.save()
        return

    # Train!
    with tqdm.tqdm(total=total_steps, leave=False) as progress:
        for _batch_idx, batch in enumerate(dataloader):
            progress.update(1)
            progress.set_postfix_str("training loop not implemented")


if __name__ == "__main__":
    finetune()
