import torch
from attn_ft.data import AttnBatch

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


def attn_logits_to_probs(attn: torch.Tensor) -> torch.Tensor:
    return torch.softmax(attn, dim=-1)


def update_head_stats(head_stats, layer_idx, t2i_attn):
    if layer_idx not in head_stats:
        num_heads = t2i_attn.shape[0]
        device = t2i_attn.device
        head_stats[layer_idx] = {
            "mass_sum": torch.zeros(num_heads, device=device),
            "entropy_sum": torch.zeros(num_heads, device=device),
            "count": torch.zeros(num_heads, device=device),
        }

    mass_per_head = t2i_attn.sum(dim=(-1, -2))

    p = t2i_attn / (t2i_attn.sum(dim=-1, keepdim=True) + 1e-8)
    entropy_per_token = -(p * torch.log(p + 1e-8)).sum(dim=-1)
    entropy_per_head = entropy_per_token.mean(dim=-1)

    head_stats[layer_idx]["mass_sum"] += mass_per_head
    head_stats[layer_idx]["entropy_sum"] += entropy_per_head
    head_stats[layer_idx]["count"] += 1


def select_grounding_heads(head_stats, top_mass_pct=10.0, low_entropy_pct=10.0):
    candidates = []
    for layer_idx, stats in head_stats.items():
        mass_mean = stats["mass_sum"] / stats["count"].clamp_min(1)
        entropy_mean = stats["entropy_sum"] / stats["count"].clamp_min(1)
        for head_idx in range(mass_mean.shape[0]):
            candidates.append(
                {
                    "layer_idx": int(layer_idx),
                    "head_idx": int(head_idx),
                    "mass": float(mass_mean[head_idx].item()),
                    "entropy": float(entropy_mean[head_idx].item()),
                }
            )

    if not candidates:
        return {}, {"num_candidates": 0, "selected": 0}

    mass_values = torch.tensor([c["mass"] for c in candidates])
    entropy_values = torch.tensor([c["entropy"] for c in candidates])

    mass_thresh = torch.quantile(mass_values, 1.0 - top_mass_pct / 100.0).item()
    entropy_thresh = torch.quantile(entropy_values, low_entropy_pct / 100.0).item()

    selected = [
        c for c in candidates
        if c["mass"] >= mass_thresh and c["entropy"] <= entropy_thresh
    ]

    selected_map = {}
    for c in selected:
        selected_map.setdefault(c["layer_idx"], []).append(c["head_idx"])

    for layer_idx in selected_map:
        selected_map[layer_idx] = sorted(selected_map[layer_idx])

    debug = {
        "num_candidates": len(candidates),
        "selected": len(selected),
        "mass_thresh": mass_thresh,
        "entropy_thresh": entropy_thresh,
    }
    return selected_map, debug



class AttnHookManager:
    def __init__(self):
        self.attentions = {}
        self.hooks = []
        self.selected_heads_map = None
        
    def _hook_fn(self, layer_idx: int):
        """Internal hook to capture the second element of the layer output."""
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 3 and output[2] is not None:
                attn = output[2]
                if self.selected_heads_map is not None:
                    selected_heads = self.selected_heads_map.get(layer_idx, [])
                    attn = attn[:, selected_heads, :, :]
                self.attentions[layer_idx] = attn
        return hook

    def attach(self, model: torch.nn.Module, selected_heads_map=None):
        """Registers hooks to all layers in the Qwen language model backbone."""
        self.remove_hooks()
        self.clear() # Ensure no stale data
        self.selected_heads_map = selected_heads_map
        
        layers = model.model.model.language_model.layers # qwen3-vl specific path to transformer layers
        for i, layer in enumerate(layers):
            # Registering to the self_attn submodule
            handle = layer.self_attn.register_forward_hook(self._hook_fn(i))
            self.hooks.append(handle)
            
        print(f"Attached {len(self.hooks)} hooks to model layers.")

    def get_attentions(self) -> list[torch.Tensor]:
        """Returns a list of attention tensors ordered by layer index."""
        return [self.attentions[i] for i in sorted(self.attentions.keys())]

    def clear(self):
        """Clears stored attention tensors."""
        self.attentions.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []