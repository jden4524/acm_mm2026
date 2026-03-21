from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
import sys
import attn_ft.modeling_qwen3_vl

# swap the original module with the modified one
sys.modules['transformers.models.qwen3_vl.modeling_qwen3_vl'] = attn_ft.modeling_qwen3_vl

import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from attn_ft.attn_hooks import AttnHookManager


def load_model_and_processor(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str],
) -> Tuple[torch.nn.Module, Any]:
    quantization_config = None
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if "qwen3-vl" in model_name.lower():
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            attn_implementation="eager",
        )

    model.config.output_attentions = True
    model.config.return_dict = True

    if lora_target_modules:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model, processor

def filter_trainable_parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}
