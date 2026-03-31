from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
import sys
import attn_ft.modeling_qwen3_vl
import attn_ft.modeling_llama

# swap the original module with the modified one
sys.modules['transformers.models.qwen3_vl.modeling_qwen3_vl'] = attn_ft.modeling_qwen3_vl
sys.modules['transformers.models.llama.modeling_llama'] = attn_ft.modeling_llama

import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from attn_ft.attn_hooks import AttnHookManager


def load_model_and_processor(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str],
    lora_layer_ids: Optional[list[int]] = None,
) -> Tuple[torch.nn.Module, Any]:

    if "qwen3-vl" in model_name.lower():
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1024 * 28 * 28
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True,min_pixels=min_pixels, max_pixels=max_pixels)

    elif "minicpm" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.config.output_attentions = True
    model.config.return_dict = True

    if lora_target_modules:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            layers_to_transform=lora_layer_ids,
            layers_pattern=r"(?:language_model|model)\.layers",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model, processor

def filter_trainable_parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}
