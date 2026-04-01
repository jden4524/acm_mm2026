from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode, resize

from attn_ft.minicpm_ds import conversation_to_ids_llama3, conversation_to_ids_minicpm


def _find_phrase_span(caption: str, phrase: str) -> Optional[Tuple[int, int]]:
    caption_cmp = caption.lower()
    phrase_cmp = phrase.lower()
    start = caption_cmp.find(phrase_cmp)
    return None if start < 0 else (start, start + len(phrase_cmp))


def _token_span_from_offsets(offsets: List[Tuple[int, int]], span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    start_char, end_char = span
    token_indices = [
        idx
        for idx, (s, e) in enumerate(offsets)
        if s != e and not (e <= start_char or s >= end_char)
    ]
    return None if not token_indices else (token_indices[0], token_indices[-1])


def _resize_mask_to_grid(mask: Image.Image, grid_size: Tuple[int, int]) -> torch.Tensor:
    h, w = grid_size
    resized = resize(mask, [h, w], interpolation=InterpolationMode.NEAREST)
    arr = np.array(resized, dtype=np.float32)
    return torch.from_numpy((arr > 0).astype(np.float32))


def _build_labels_from_context(
    input_ids: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[context != 0] = -100
    return labels


def _assistant_content_start(context: torch.Tensor) -> Optional[int]:
    assistant_positions = (context == 0).nonzero(as_tuple=False).squeeze(-1)
    if assistant_positions.numel() == 0:
        return None
    return int(assistant_positions[0].item())


def _compute_image_bound(input_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[0] + 1
    image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[0]
    valid_image_nums = min(len(image_start_tokens), len(image_end_tokens))
    if valid_image_nums == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.stack(
        [image_start_tokens[:valid_image_nums], image_end_tokens[:valid_image_nums]],
        dim=-1,
    ).long()


def _image_token_indices_from_bounds(image_bound: torch.Tensor) -> torch.Tensor:
    if image_bound.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    ranges = [
        torch.arange(int(start), int(end), dtype=torch.long)
        for start, end in image_bound.tolist()
        if int(end) > int(start)
    ]
    if not ranges:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(ranges, dim=0)


def _minicpm_llm_type(model_name: str) -> str:
    return "llama3" if "llama3" in model_name.lower() else "minicpm"


def _best_factorized_grid(num_tokens: int, image_size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = image_size
    target_ratio = width / max(height, 1)
    best_grid = (1, num_tokens)
    best_error = float("inf")

    for height_tokens in range(1, int(num_tokens ** 0.5) + 1):
        if num_tokens % height_tokens != 0:
            continue
        width_tokens = num_tokens // height_tokens
        error = abs((width_tokens / height_tokens) - target_ratio)
        if error < best_error:
            best_grid = (height_tokens, width_tokens)
            best_error = error

    return best_grid


def _encode_minicpm_conversation(
    conversation: List[Dict[str, str]],
    tokenizer: Any,
    llm_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if llm_type == "llama3":
        raw_input_ids, raw_context, _ = conversation_to_ids_llama3(conversation, tokenizer)
    else:
        raw_input_ids, raw_context, _ = conversation_to_ids_minicpm(conversation, tokenizer)

    input_ids = torch.as_tensor(np.asarray(raw_input_ids, dtype=np.int64))
    context = torch.as_tensor(np.asarray(raw_context, dtype=np.int8))
    return input_ids.long(), context


def _build_minicpm_prompt(question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": f"{question}(<image>./</image>)\n"},
        {"role": "assistant", "content": answer},
    ]


def _slice_mask_for_minicpm(mask: Image.Image, image: Image.Image, image_processor: Any) -> torch.Tensor:
    query_tokens = int(getattr(image_processor, "image_feature_size", 64))

    source_image, patches, best_grid = image_processor.slice_image(
        image,
        image_processor.max_slice_nums,
        image_processor.scale_resolution,
        image_processor.patch_size,
    )

    flattened_masks = [
        _resize_mask_to_grid(
            mask.resize(source_image.size, resample=Image.Resampling.NEAREST),
            _best_factorized_grid(query_tokens, source_image.size),
        ).reshape(-1)
    ]

    if best_grid is not None and len(patches) > 0:
        refine_size = image_processor.get_refine_size(
            image.size,
            best_grid,
            image_processor.scale_resolution,
            image_processor.patch_size,
            allow_upscale=True,
        )
        refine_mask = mask.resize(refine_size, resample=Image.Resampling.NEAREST)
        mask_patches = image_processor.split_to_patches(refine_mask, best_grid)
        for row, image_row in zip(mask_patches, patches):
            for patch, image_patch in zip(row, image_row):
                flattened_masks.append(
                    _resize_mask_to_grid(
                        patch,
                        _best_factorized_grid(query_tokens, image_patch.size),
                    ).reshape(-1)
                )

    return torch.cat(flattened_masks, dim=0)


def _build_minicpm_patch_masks(mask: Image.Image, image: Image.Image, image_processor: Any) -> List[torch.Tensor]:
    patch_size = int(image_processor.patch_size)

    source_image, patches, best_grid = image_processor.slice_image(
        image,
        image_processor.max_slice_nums,
        image_processor.scale_resolution,
        image_processor.patch_size,
    )

    patch_masks = [
        _resize_mask_to_grid(
            mask.resize(source_image.size, resample=Image.Resampling.NEAREST),
            (source_image.size[1] // patch_size, source_image.size[0] // patch_size),
        ).reshape(-1)
    ]

    if best_grid is not None and len(patches) > 0:
        refine_size = image_processor.get_refine_size(
            image.size,
            best_grid,
            image_processor.scale_resolution,
            image_processor.patch_size,
            allow_upscale=True,
        )
        refine_mask = mask.resize(refine_size, resample=Image.Resampling.NEAREST)
        mask_patches = image_processor.split_to_patches(refine_mask, best_grid)
        for row in mask_patches:
            for patch in row:
                patch_masks.append(
                    _resize_mask_to_grid(
                        patch,
                        (patch.size[1] // patch_size, patch.size[0] // patch_size),
                    ).reshape(-1)
                )

    return patch_masks


def _pad_minicpm_examples(examples: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    return {
        "input_ids": pad_sequence(
            [example["input_ids"].long() for example in examples],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "position_ids": pad_sequence(
            [example["position_ids"].long() for example in examples],
            batch_first=True,
            padding_value=0,
        ),
        "attention_mask": pad_sequence(
            [example["attention_mask"].bool() for example in examples],
            batch_first=True,
            padding_value=0,
        ),
        "pixel_values": [example["pixel_values"] for example in examples],
        "tgt_sizes": [example["tgt_sizes"] for example in examples],
        "image_bound": [example["image_bound"] for example in examples],
    }




@dataclass
class AttnBatch:
    inputs: Dict[str, Any]
    masks: List[torch.Tensor]
    token_spans: List[Optional[slice]]
    image_token_indices: List[torch.Tensor]
    valid_supervision_indices: List[int]
    answers: List[str]
    vision_patch_masks: Optional[List[List[torch.Tensor]]] = None
    # image_stems: List[Optional[str]]
    labels: Optional[torch.Tensor] = None


class AttnSupervisionCollator:
    def __init__(
        self,
        processor: Optional[Any],
        model_name: str,
    ) -> None:
        self.processor = processor
        self.model_name = model_name

    def __call__(self, batch: List[Dict[str, Any]]) -> AttnBatch:
        if "qwen" in self.model_name.lower():
            return self._collate_qwen(batch)
        if "minicpm" in self.model_name.lower():
            return self._collate_minicpm(batch)
        raise ValueError(f"Unsupported model for collation: {self.model_name}")

    def _collate_qwen(self, batch: List[Dict[str, Any]]) -> AttnBatch:
        images = [b["image"] for b in batch]
        masks_list = [b["mask"] for b in batch]
        phrases_list = [b["phrase"] for b in batch]

        answers = [str(sample["answer"]) for sample in batch]
        questions = [str(sample["question"]) for sample in batch]

        prompts = []
        for q, a, img in zip(questions, answers, images):
            message = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": q},
                        ],
                    },
                        {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": a},
                        ],
                    }]
            prompt = self.processor.apply_chat_template(
                message,
                add_generation_prompt=False,
                tokenize=False,
            )
            prompts.append(prompt)

        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        tokenized = self.processor.tokenizer(
            answers,
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=False,
        )

        offsets = tokenized["offset_mapping"]
        assistant_idx = (inputs["input_ids"] == 77091).nonzero(as_tuple=True)[1].tolist()
        img_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_token_indices = [
            (inputs["input_ids"][i] == img_token_id).nonzero(as_tuple=False).squeeze(1)
            for i in range(len(batch))
        ]
        # calculate vision grid size based on processor/model config and image sizes
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        bsz = len(batch)

        masks = [] #torch.zeros((bsz, h, w), dtype=torch.float32)

        token_spans = []
        valid_supervision_indices = []
        labels = inputs["input_ids"].clone()

        for i in range(bsz):
            answer = answers[i]
            mask_img = masks_list[i]
            phrase = phrases_list[i]
            vision_grid_size = (inputs.image_grid_thw[i,1:] / 2).int().tolist() # _infer_vision_grid_size(self.processor, img_size=images[i].size)
            resized = _resize_mask_to_grid(mask_img, vision_grid_size)
            masks.append(resized)
            
            labels[i, :assistant_idx[i]+2] = -100
            padding_mask = (inputs["input_ids"][i] == pad_token_id)
            labels[i][padding_mask] = -100

            span = _find_phrase_span(answer, phrase)
            if span is None:
                token_spans.append(None)
                continue
            token_span = _token_span_from_offsets(offsets[i], span)
            if token_span is None:
                token_spans.append(None)
                continue
            # assistant token is followed by \n, so starting from +2
            token_spans.append(slice(token_span[0]+assistant_idx[i]+2, token_span[1]+assistant_idx[i]+3))
            if image_token_indices[i].numel() > 0:
                valid_supervision_indices.append(i)

        return AttnBatch(
            inputs=inputs,
            masks=masks,
            token_spans=token_spans,
            image_token_indices=image_token_indices,
            valid_supervision_indices=valid_supervision_indices,
            answers=answers,
            labels=labels
        )

    def _collate_minicpm(self, batch: List[Dict[str, Any]]) -> AttnBatch:
        tokenizer = self.processor.tokenizer
        image_processor = self.processor.image_processor
        llm_type = _minicpm_llm_type(self.model_name)

        answers = [str(sample["answer"]) for sample in batch]
        questions = [str(sample["question"]) for sample in batch]
        phrases_list = [str(sample["phrase"]) for sample in batch]
        images = [sample["image"] for sample in batch]
        masks_list = [sample["mask"] for sample in batch]

        tokenized_answers = tokenizer(
            answers,
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=False,
        )
        offsets = tokenized_answers["offset_mapping"]

        sample_inputs: List[Dict[str, Any]] = []
        masks: List[torch.Tensor] = []
        token_spans: List[Optional[slice]] = []
        image_token_indices: List[torch.Tensor] = []
        valid_supervision_indices: List[int] = []
        labels_list: List[torch.Tensor] = []
        vision_patch_masks: List[List[torch.Tensor]] = []

        for batch_idx, (image, mask_img, phrase, question, answer) in enumerate(
            zip(images, masks_list, phrases_list, questions, answers)
        ):
            conversation = _build_minicpm_prompt(question, answer)
            resolved_conversation = copy.deepcopy(conversation)
            resolved_conversation[0]["content"] = resolved_conversation[0]["content"].replace(
                "(<image>./</image>)",
                image_processor.get_slice_image_placeholder(image.size),
            )

            input_ids, context = _encode_minicpm_conversation(
                resolved_conversation,
                tokenizer,
                llm_type,
            )
            assistant_start = _assistant_content_start(context)
            image_bound = _compute_image_bound(input_ids, tokenizer)
            image_indices = _image_token_indices_from_bounds(image_bound)
            image_features = image_processor.preprocess(image, return_tensors="pt")
            labels = _build_labels_from_context(input_ids, context)

            sample_inputs.append(
                {
                    "input_ids": input_ids,
                    "position_ids": torch.arange(input_ids.size(0), dtype=torch.long),
                    "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
                    "pixel_values": image_features["pixel_values"][0],
                    "tgt_sizes": image_features["tgt_sizes"][0].long(),
                    "image_bound": image_bound,
                }
            )
            labels_list.append(labels.long())
            image_token_indices.append(image_indices)
            masks.append(_slice_mask_for_minicpm(mask_img, image, image_processor))
            vision_patch_masks.append(_build_minicpm_patch_masks(mask_img, image, image_processor))

            span = _find_phrase_span(answer, phrase)
            token_span = _token_span_from_offsets(offsets[batch_idx], span) if span is not None else None
            if token_span is None or assistant_start is None:
                token_spans.append(None)
                continue

            token_spans.append(slice(assistant_start + token_span[0], assistant_start + token_span[1] + 1))
            if image_indices.numel() > 0:
                valid_supervision_indices.append(batch_idx)

        inputs = _pad_minicpm_examples(sample_inputs, tokenizer.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return AttnBatch(
            inputs=inputs,
            masks=masks,
            token_spans=token_spans,
            image_token_indices=image_token_indices,
            valid_supervision_indices=valid_supervision_indices,
            answers=answers,
            vision_patch_masks=vision_patch_masks,
            labels=labels,
        )
