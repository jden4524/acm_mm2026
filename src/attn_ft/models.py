from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
import sys
import attn_ft.modeling_qwen3_vl
import attn_ft.modeling_llama
import attn_ft.modeling_mllama

# swap the original module with the modified one
sys.modules['transformers.models.qwen3_vl.modeling_qwen3_vl'] = attn_ft.modeling_qwen3_vl
sys.modules['transformers.models.llama.modeling_llama'] = attn_ft.modeling_llama
sys.modules['transformers.models.mllama.modeling_mllama'] = attn_ft.modeling_mllama

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from attn_ft.attn_hooks import AttnHookManager


def _patch_minicpm_forward(model: torch.nn.Module) -> None:
    model_cls = type(model)
    if getattr(model_cls, "_attn_ft_forward_patched", False):
        return

    original_forward = model_cls.forward

    def patched_forward(self, data, **kwargs):
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("position_ids", None)
        return original_forward(self, data, **kwargs)

    model_cls.forward = patched_forward
    model_cls._attn_ft_forward_patched = True


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

    elif "llama-3.2" in model_name.lower() and "vision" in model_name.lower():
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        processor.image_processor.patch_size = model.config.vision_config.patch_size
        processor.image_processor.num_patches = (
            (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2 + 1
        )

    elif "minicpm" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        _patch_minicpm_forward(model)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.config.output_attentions = "llama-3.2" not in model_name.lower()
    model.config.return_dict = True

    if lora_target_modules:
        lora_kwargs = {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": lora_target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if lora_layer_ids is not None:
            lora_kwargs["layers_to_transform"] = lora_layer_ids
            lora_kwargs["layers_pattern"] = r"(?:language_model|model)\.layers"

        lora_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, lora_config)

    return model, processor

def filter_trainable_parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}
