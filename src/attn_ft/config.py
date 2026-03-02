from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    hf_dataset_id: str
    split: str
    caption_field: str
    image_field: str
    mask_root: str
    max_samples: Optional[int]


@dataclass
class ModelConfig:
    name: str
    trust_remote_code: bool
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    attention_layers: Dict[int, float]
    attention_heads: List[int]


@dataclass
class TrainConfig:
    run_name: str
    loss: str
    loss_weight: float
    seed: int
    micro_batch_size: int
    effective_batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    warmup_steps: int
    mixed_precision: str
    log_every: int
    save_every: int
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
def load_config(path: str | Path) -> Config:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text())

    dset = raw.get("dataset", {})
    model = raw.get("model", {})
    train = raw.get("train", {})

    dataset_cfg = DatasetConfig(
        hf_dataset_id=dset.get("hf_dataset_id", ""),
        split=dset.get("split", ""),
        caption_field=dset.get("caption_field", "caption"),
        image_field=dset.get("image_field", "image"),
        mask_root=dset.get("mask_root", ""),
        max_samples=dset.get("max_samples")
    )

    model_cfg = ModelConfig(
        name=model.get("name", ""),
        trust_remote_code=model.get("trust_remote_code", False),
        load_in_4bit=model.get("load_in_4bit", False),
        lora_r=model.get("lora_r", 8),
        lora_alpha=model.get("lora_alpha", 16),
        lora_dropout=model.get("lora_dropout", 0.05),
        lora_target_modules=model.get("lora_target_modules", []),
        attention_layers=model.get("attention_layers", {-2:1.0}),
        attention_heads=model.get("attention_heads", []),
    )

    train_cfg = TrainConfig(
        run_name=train.get("run_name"),
        loss=train.get("loss", "ce"),
        loss_weight=train.get("loss_weight", 1.0),
        seed=train.get("seed", 42),
        micro_batch_size=train.get("micro_batch_size", 1),
        effective_batch_size=train.get("effective_batch_size", 4),
        num_epochs=train.get("num_epochs", 1),
        lr=train.get("lr", 1.0e-4),
        weight_decay=train.get("weight_decay", 0.0),
        warmup_steps=train.get("warmup_steps", 0),
        mixed_precision=train.get("mixed_precision", "no"),
        log_every=train.get("log_every", 10),
        save_every=train.get("save_every", 500),
        wandb_enabled=train.get("wandb_enabled", False),
        wandb_project=train.get("wandb_project", "attn_ft"),
        wandb_entity=train.get("wandb_entity"),
        wandb_run_name=train.get("wandb_run_name"),
    )

    return Config(dataset=dataset_cfg, model=model_cfg, train=train_cfg)
