"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""

import os
from typing import Tuple

from torch.utils.data import Dataset
from pathlib import Path

from latentvla.data_provider.datasets import RLDSDataset
from latentvla.data_provider.data_utils import PaddedCollatorForActionPrediction, RLDSBatchTransform, RLDSBatchTransformLatentAction, RLDSBatchTransformInternVL
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from latentvla.models.laq_model import LatentActionQuantization

def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    default_image_resolution: Tuple[int, int],
    latent_model: LatentActionQuantization,
    padding_side: str = "right",
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    image_aug: bool = False,
    window_size: int = 8,
    goal_image_step: int = 8,
    model_id: str = "Qwen2.5-VL-3B-Instruct",
) -> Tuple[Dataset, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    if model_id == "Qwen2.5-VL-3B-Instruct":
        batch_transform = RLDSBatchTransformLatentAction(
            processor=processor, latent_model=latent_model, window_size=window_size)
    elif model_id == "InternVL3_5-1B" or model_id == "InternVL3_5-2B":
        batch_transform = RLDSBatchTransformInternVL(
            vlm=model, tokenizer=tokenizer, latent_model=latent_model, window_size=window_size)
    
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        window_size=window_size,
        goal_image_step=goal_image_step,
    )

    return dataset, collator