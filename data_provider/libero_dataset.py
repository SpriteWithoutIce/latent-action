"""Utilities for loading the modified LIBERO Spatial dataset.

This module provides a thin convenience wrapper around :class:`RLDSDataset`
that resolves the on-disk path of the ``libero_spatial_no_noops`` dataset and
initialises the same transformation / collation objects that were used in the
original OpenVLA codebase.

The default dataset location matches the directory structure provided by the
user:

``/home/linyihan/linyh/datasets/modified_libero_rlds/libero_spatial_no_noops/1.0.0``

If your dataset is stored elsewhere simply pass ``data_root_dir`` when calling
``create_libero_spatial_no_noops_dataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from latentvla.data_provider.data_utils import (
    PaddedCollatorForActionPrediction,
    RLDSBatchTransformInternVL,
)
from latentvla.data_provider.datasets import RLDSDataset

# Dataset metadata -----------------------------------------------------------------

DEFAULT_LIBERO_DATASET_NAME = "libero_spatial_no_noops"
DEFAULT_LIBERO_DATA_ROOT = Path(
    "/home/linyihan/linyh/datasets/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
)


def _resolve_libero_builder_dir(
    data_root_dir: Path, dataset_name: str = DEFAULT_LIBERO_DATASET_NAME
) -> Path:
    """Locate the TFDS builder directory containing ``dataset_info.json``.

    The helper is tolerant to three common layouts:

    * ``data_root_dir`` already points at the version directory (``.../1.0.0``).
    * ``data_root_dir`` points at the dataset directory
      (``.../libero_spatial_no_noops``).
    * ``data_root_dir`` is the parent directory that contains multiple dataset
      folders.

    Args:
        data_root_dir: Base path provided by the user.
        dataset_name: Dataset identifier used within the RLDS config registry.

    Returns:
        Path to the directory that should be passed to ``tfds.builder_from_directory``.

    Raises:
        FileNotFoundError: If a matching TFDS builder directory could not be located.
    """

    candidate_dirs = []
    root = data_root_dir.expanduser()
    candidate_dirs.append(root)

    # Allow passing the parent directory that contains dataset sub-folders.
    candidate_dirs.append(root / dataset_name)

    # If ``root`` already points at ``.../<dataset_name>`` also search the parent.
    if root.name == dataset_name:
        candidate_dirs.append(root.parent / dataset_name)

    for candidate in candidate_dirs:
        if not candidate.exists():
            continue

        dataset_info = candidate / "dataset_info.json"
        if dataset_info.exists():
            return candidate

        # Search for versioned sub-directories (e.g., ``1.0.0``).
        for subdir in sorted(candidate.iterdir()):
            if subdir.is_dir() and (subdir / "dataset_info.json").exists():
                return subdir

    raise FileNotFoundError(
        f"Could not locate a TFDS builder directory for '{dataset_name}' under "
        f"{data_root_dir}. Ensure the dataset has been materialised correctly."
    )


def create_libero_spatial_no_noops_dataset(
    action_tokenizer,
    tokenizer,
    *,
    data_root_dir: Optional[Path] = None,
    window_size: int = 8,
    image_size: int = 256,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    image_aug: bool = True,
    use_wrist_image: bool = True,
    use_proprio: bool = True,
    goal_image_step: Optional[int] = None,
) -> Tuple[RLDSDataset, PaddedCollatorForActionPrediction]:
    """Instantiate the RLDS pipeline for the LIBERO spatial training set.

    Args:
        action_tokenizer: Tokeniser used for discretising low-level actions.
        tokenizer: Text tokenizer from the language backbone.
        data_root_dir: Override for the dataset location. Defaults to
            ``DEFAULT_LIBERO_DATA_ROOT``.
        window_size: Temporal window used when chunking trajectories.
        image_size: Square image resolution to resize RGB observations to.
        shuffle_buffer_size: Size of the shuffling buffer used by RLDS.
        train: Whether to load the training split (``True``) or validation split.
        image_aug: Enable image augmentations within the RLDS pipeline.
        use_wrist_image: Include wrist camera observations in the batch transform.
        use_proprio: Include proprioceptive state in the batch transform.
        goal_image_step: Step offset for sampling goal images. Defaults to the
            provided ``window_size`` when ``None``.

    Returns:
        Tuple of ``(dataset, collator)`` ready to be consumed by a PyTorch
        ``DataLoader``.
    """

    resolved_root = _resolve_libero_builder_dir(
        Path(data_root_dir) if data_root_dir is not None else DEFAULT_LIBERO_DATA_ROOT
    )

    batch_transform = RLDSBatchTransformInternVL(
        action_tokenizer,
        tokenizer,
        use_wrist_image=use_wrist_image,
        use_proprio=use_proprio,
    )

    dataset = RLDSDataset(
        resolved_root,
        DEFAULT_LIBERO_DATASET_NAME,
        batch_transform,
        resize_resolution=(image_size, image_size),
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        window_size=window_size,
        goal_image_step=goal_image_step if goal_image_step is not None else window_size,
    )

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side="right",
    )

    return dataset, collator


__all__ = [
    "DEFAULT_LIBERO_DATA_ROOT",
    "DEFAULT_LIBERO_DATASET_NAME",
    "create_libero_spatial_no_noops_dataset",
]

