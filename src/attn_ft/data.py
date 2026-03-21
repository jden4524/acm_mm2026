from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode, resize

def _extract_caption(
    value: Any,
    phrases: Optional[List[str]] = None,
) -> str:
    if isinstance(value, dict):
        value = value.get("raw", value)

    if isinstance(value, list) and value:
        if phrases:
            phrases_cmp = [p.lower() for p in phrases]
            for caption in value:
                caption_str = str(caption)
                caption_cmp = caption_str.lower()
                if any(phrase in caption_cmp for phrase in phrases_cmp):
                    return caption_str
        value = value[0]

    return str(value)


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

class Flickr30kSamDataset(Dataset):
    def __init__(
        self,
        hf_dataset_id: str,
        split: str,
        caption_field: str,
        image_field: str,
        mask_root: str,
        max_samples: Optional[int] = None,
    ) -> None:
        self.dataset = load_dataset(hf_dataset_id, split=split, trust_remote_code=True)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))

        self.caption_field = caption_field
        self.image_field = image_field
        self.mask_root = Path(mask_root)
        self.flattened_map: List[Tuple[int, Path, str]] = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            image = sample[self.image_field]
            filename = sample.get("filename") or getattr(image, "filename", None)
            if not filename:
                continue
            image_stem = Path(filename).stem
            folder = self.mask_root / image_stem
            if not folder.exists():
                continue
            mask_files = sorted(
                p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            for mask_path in mask_files:
                self.flattened_map.append((idx, mask_path, image_stem))

    def __len__(self) -> int:
        return len(self.flattened_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx, mask_path, image_stem = self.flattened_map[idx]
        sample = self.dataset[dataset_idx]
        image = sample[self.image_field]
        mask = Image.open(mask_path).convert("L")
        phrase = mask_path.stem
        caption = _extract_caption(
            sample[self.caption_field],
            phrases=[phrase],
        )

        return {
            "image": image,
            "caption": caption,
            "mask": mask,
            "phrase": phrase,
            "image_stem": image_stem,
        }


@dataclass
class AttnBatch:
    inputs: Dict[str, torch.Tensor]
    masks: List[torch.Tensor]
    token_spans: List[Optional[slice]]
    image_token_indices: List[torch.Tensor]
    valid_supervision_indices: List[int]
    captions: List[str]
    image_stems: List[Optional[str]]
    labels: Optional[torch.Tensor] = None


class AttnSupervisionCollator:
    def __init__(
        self,
        processor: Optional[Any]
    ) -> None:
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> AttnBatch:
        images = [b["image"] for b in batch]
        captions = [b["caption"] for b in batch]
        masks_list = [b["mask"] for b in batch]
        phrases_list = [b["phrase"] for b in batch]
        image_stems = [b["image_stem"] for b in batch]


        prompts = []
        for cap, img in zip(captions, images):
            message = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "What is in this image?"},
                        ],
                    },
                        {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": cap},
                        ],
                    }]
            prompt = self.processor.apply_chat_template(message,
                    add_generation_prompt=False,
                    tokenize=False)
            prompts.append(prompt)
                
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)

        tokenized = self.processor.tokenizer(
            captions,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
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
            caption = captions[i]
            mask_img = masks_list[i]
            phrase = phrases_list[i]
            vision_grid_size = (inputs.image_grid_thw[i,1:] / 2).int().tolist() # _infer_vision_grid_size(self.processor, img_size=images[i].size)
            resized = _resize_mask_to_grid(mask_img, vision_grid_size)
            masks.append(resized)
            
            labels[i, :assistant_idx[i]+2] = -100
            padding_mask = (inputs["input_ids"][i] == pad_token_id)
            labels[i][padding_mask] = -100

            span = _find_phrase_span(caption, phrase)
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
            captions=captions,
            image_stems=image_stems,
            labels=labels
        )
