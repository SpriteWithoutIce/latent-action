"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

import os
import re
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, Any, Dict, Type, Callable, Optional, Any

import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import tensorflow as tf

from PIL import Image
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
import torchvision.transforms.functional as F
from torchvision import transforms
NUM_TOKENS = 64
IGNORE_INDEX = -100
PROPRIO_DIM = 8


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        latent_action_idx = [instance["latent_action_idx"] for instance in instances]
        dataset_names = [inst["dataset_name"] for inst in instances] if "dataset_name" in instances[0] else None

        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        
        attention_mask = input_ids.ne(self.pad_token_id)

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            proprio=proprio,
            attention_mask=attention_mask,
            actions=actions,
            labels=labels,
            latent_action_idx=latent_action_idx
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output 

def prepare_input_ids(prompt_builder, conversation, print_prompt_limit, tokenizer, num_answer_tokens, predict_stop_token):
    for turn in conversation:
        prompt_builder.add_turn(turn["from"], turn["value"])

    if print_prompt_limit > 0:
        print("Conversation:", conversation)
        p = prompt_builder.get_prompt()
        print("Prompt:", p)

    # Tokenize (w/ `base_tokenizer`)
    input_ids = tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
    labels = list(input_ids)

    # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
    input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    
    # critical, some tokenizers have different numbers of "end tokens".
    num_end_tokens = 1
    if isinstance(tokenizer, Qwen2TokenizerFast):
        # Qwen has <|im_end|><|endoftext|> for example
        num_end_tokens = 2

    # mask the input id tokens parts
    labels[0 : -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
    if not predict_stop_token:
        labels[-num_end_tokens:] = IGNORE_INDEX

    return input_ids, labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

num_image_token = 256

IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"
from transformers import AutoTokenizer
from data_provider.action_tokenizer import ActionTokenizer
from latent_action_model.genie.model import ControllableDINOLatentActionModel
@dataclass
class RLDSBatchTransformInternVL:
    action_tokenizer: ActionTokenizer
    tokenizer: AutoTokenizer
    latent_action_tokenizer: ControllableDINOLatentActionModel
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, actions = rlds_batch["dataset_name"], rlds_batch["action"]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["goal_image"]["image_primary"])

        # Get latent action tokens
        with torch.no_grad():
            initial_pixel_values = transforms.ToTensor()(img)
            target_pixel_values= transforms.ToTensor()(img_k)

            video = torch.stack([initial_pixel_values, target_pixel_values], dim=0).unsqueeze(0).to(self.latent_action_tokenizer.device)
            latent_action_idx = self.latent_action_tokenizer.vq_encode(video)['indices'].squeeze()

        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        pixel_values = load_image(img).to(torch.bfloat16).cuda()
        
        action_chunk_string = self.action_tokenizer(actions, use_minivlm=True)
        flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]
        action_chunk_len = len(flattened_action_chunk_string) 
        prompt_text = f"What action should the robot take to {lang}?"

        input_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        # print("input_ids: ", input_ids.shape)
        if NUM_TOKENS<len(flattened_action_chunk_string):
            input_ids = input_ids.tolist() + flattened_action_chunk_string[:NUM_TOKENS]
        else:
            remaining_length = NUM_TOKENS - len(flattened_action_chunk_string)
            extended_array = random.choices(flattened_action_chunk_string, k=remaining_length)
            
            input_ids = input_ids.tolist() + flattened_action_chunk_string + extended_array
        action_chunk_len = NUM_TOKENS
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # print("labels: ",labels.shape)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        return_dict = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            actions=actions,
            labels=labels,
            dataset_name=dataset_name,
            latent_action_idx=latent_action_idx
        )
        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = load_image(img_wrist).to(torch.bfloat16).cuda()
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            # libero 这里会变成9维，6+1+2，有一个是0
            if proprio.shape[-1] != PROPRIO_DIM:
                proprio = tf.concat([proprio[:, :6], proprio[:, 7:]], axis=1)
            return_dict["proprio"] = proprio

        return return_dict