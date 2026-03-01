from typing import Dict, List, Optional
from attn_ft.data import AttnBatch
import torch

def extract_t2i_attn(
    attn: torch.Tensor,
    batch: AttnBatch,
    processor
) -> torch.Tensor:
    """Extracts and aggregates text-to-image attention maps

    Expects a tensor with a head dimension (e.g. [B, H, T, S] or [H, T, S]).
    Returns a list of tensors, each with shape (heads, num_text_tokens, H, W).
    """
    img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    extracted = []
    for b in range(attn.shape[0]):
        is_image = batch.inputs["input_ids"][b] == img_token_id
        image_idx = is_image.nonzero(as_tuple=False).squeeze(1)
        layer_attn = attn[b]  # [heads, seq, seq]
        if batch.token_spans[b]:
            text_to_image = layer_attn[:, batch.token_spans[b], image_idx]
        
            # H, W = batch.masks[b].shape # 
            # attn_2d = text_to_image.view(layer_attn.shape[0], -1, H, W) # Reshape to (heads, num_text_tokens, H, W)
        else:
            text_to_image = None
        extracted.append(text_to_image)

    return extracted



class AttnHookManager:
    def __init__(self):
        self.attentions: Dict[int, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
    def _hook_fn(self, layer_idx: int):
        """Internal hook to capture the second element of the layer output."""
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                self.attentions[layer_idx] = output[2]
        return hook

    def attach(self, model: torch.nn.Module):
        """Registers hooks to all layers in the Qwen language model backbone."""
        self.clear() # Ensure no stale hooks or data
        
        layers = model.model.model.language_model.layers # qwen3-vl specific path to transformer layers
        for i, layer in enumerate(layers):
            # Registering to the self_attn submodule
            handle = layer.self_attn.register_forward_hook(self._hook_fn(i))
            self.hooks.append(handle)
            
        print(f"Attached {len(self.hooks)} hooks to model layers.")

    def get_attentions(self) -> List[torch.Tensor]:
        """Returns a list of attention tensors ordered by layer index."""
        return [self.attentions[i] for i in sorted(self.attentions.keys())]

    def clear(self):
        """Removes all hooks and clears stored tensors to free VRAM."""
        self.attentions.clear()