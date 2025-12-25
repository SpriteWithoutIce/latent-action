"""Train the stage-2 latent action model on Libero RLDS data.

This script fine-tunes the controllable latent action model (LAM) directly,
continuing from an existing stage-2 checkpoint. It keeps the inference code in
``libero_dataset.py`` untouched by defining a dedicated training entrypoint and
data pipeline here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import draccus
import torch
import tqdm
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from data_provider.datasets import RLDSDataset
from data_provider.utils import set_global_seed
from latent_action_model.genie.modules import ControllableDINOLatentActionModel

# Avoid tokenizer parallelism warning (unused here but keeps parity with other scripts)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RLDSBatchTransformLAM:
    """Minimal RLDS transform for LAM training.

    Converts the Libero start and goal images into a two-frame video tensor.
    """

    def __init__(self, resolution: int = 224) -> None:
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, rlds_batch: Dict) -> Dict:
        start = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        goal = Image.fromarray(rlds_batch["goal_image"]["image_primary"])

        video = torch.stack([self.image_transform(start), self.image_transform(goal)], dim=0)
        instruction = rlds_batch["task"]["language_instruction"].decode().lower()

        return {
            "videos": video,
            "task_instruction": instruction,
            "dataset_name": rlds_batch["dataset_name"],
        }


class CollatorForLAM:
    """Collate Libero RLDS samples for LAM training."""

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        videos = torch.stack([instance["videos"] for instance in instances])
        dataset_names = [instance["dataset_name"] for instance in instances]
        instructions = [instance["task_instruction"] for instance in instances]

        return {
            "videos": videos,
            "dataset_names": dataset_names,
            "task_instruction": instructions,
        }


def load_lam_checkpoint(lam_path: str, device: torch.device) -> ControllableDINOLatentActionModel:
    """Load a stage-2 LAM checkpoint and return a trainable model."""

    lam_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=768,
        latent_dim=128,
        num_latents=16,
        patch_size=14,
        enc_blocks=12,
        dec_blocks=12,
        num_heads=12,
        dropout=0.0,
    )

    checkpoint = torch.load(lam_path, map_location="cpu")["state_dict"]
    remapped_state = {key.replace("lam.", ""): value for key, value in checkpoint.items()}
    lam_model.load_state_dict(remapped_state, strict=True)
    return lam_model.to(device)


def lam_loss(outputs: Dict, vq_beta: float) -> torch.Tensor:
    """Replicate the Lightning LAM stage-2 loss."""

    mse_loss = ((outputs["target"] - outputs["recon"]) ** 2).mean()
    q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
    commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

    loss = mse_loss + q_loss + vq_beta * commit_loss

    if "z_q_uncontrol" in outputs:
        q_loss_uncontrol = ((outputs["emb_uncontrol"].detach() - outputs["z_uncontrol"]) ** 2).mean()
        commit_loss_uncontrol = ((outputs["emb_uncontrol"] - outputs["z_uncontrol"].detach()) ** 2).mean()
        loss = loss + q_loss_uncontrol + vq_beta * commit_loss_uncontrol

    return loss


@dataclass
class Stage2TrainConfig:
    seed: int = 42
    lam_path: str = "/home/linyihan/linyh/latent-action/latent_action_model/checkpoints/lam-stage-2.ckpt"

    # data
    data_root_dir: str = "/home/linyihan/linyh/datasets/modified_libero_rlds"
    data_mix: List[str] = field(default_factory=lambda: ["libero_object_no_noops"])
    resize_resolution: int = 224
    goal_image_step: int = 32

    # training
    max_steps: int = 1000
    per_device_batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_dir: str = "finetuned_lam_stage2"
    save_every: Optional[int] = None
    vq_beta: float = 0.25


def build_dataloader(cfg: Stage2TrainConfig, worker_init_fn) -> DataLoader:
    transform = RLDSBatchTransformLAM(cfg.resize_resolution)
    dataset = RLDSDataset(
        Path(cfg.data_root_dir),
        cfg.data_mix,
        batch_transform=transform,
        resize_resolution=(cfg.resize_resolution, cfg.resize_resolution),
        shuffle_buffer_size=100_000,
        train=True,
        image_aug=True,
        window_size=1,
        goal_image_step=cfg.goal_image_step,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=None,
        collate_fn=CollatorForLAM(),
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )


@draccus.wrap()
def train(cfg: Stage2TrainConfig) -> None:
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lam_model = load_lam_checkpoint(cfg.lam_path, device)
    lam_model.train()

    dataloader = build_dataloader(cfg, worker_init_fn)

    optimizer = torch.optim.AdamW(
        lam_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.0 if cfg.warmup_steps > 0 else 1.0,
        total_iters=cfg.warmup_steps if cfg.warmup_steps > 0 else cfg.max_steps,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        for batch_idx, batch in enumerate(dataloader):
            videos = batch["videos"].to(device)

            with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                outputs = lam_model({"videos": videos})
                loss = lam_loss(outputs, cfg.vq_beta) / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lam_model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                completed_steps += 1
                progress.update(1)

                if completed_steps % cfg.log_every == 0:
                    progress.set_postfix(loss=float(loss.detach().cpu()))

                if cfg.save_every and completed_steps % cfg.save_every == 0:
                    save_path = Path(cfg.save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    ckpt_path = save_path / f"lam_step_{completed_steps}.ckpt"
                    torch.save({"state_dict": {f"lam.{k}": v.cpu() for k, v in lam_model.state_dict().items()}}, ckpt_path)

                if completed_steps >= cfg.max_steps:
                    break

        progress.set_postfix_str("training complete")

    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"lam.{k}": v.cpu() for k, v in lam_model.state_dict().items()}}, save_path / "lam_final.ckpt")


if __name__ == "__main__":
    train()