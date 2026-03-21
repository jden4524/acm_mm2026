import torch
from collections import defaultdict
from attn_ft.data import AttnBatch
from attn_ft.losses import soft_suppression_loss


def extract_t2i_attn(
    attn: torch.Tensor,
    batch: AttnBatch,
    processor,
) -> torch.Tensor:
    """Extracts and aggregates text-to-image attention maps.

    Expects a tensor with a head dimension (e.g. [B, H, T, S] or [H, T, S]).
    Returns a list of tensors, each with shape (heads, num_text_tokens, H, W).
    """
    img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    extracted = []
    for b, layer_attn in enumerate(attn):
        is_image = batch.inputs["input_ids"][b] == img_token_id
        image_idx = is_image.nonzero(as_tuple=False).squeeze(1)
        token_span = batch.token_spans[b]
        extracted.append(layer_attn[:, token_span, image_idx] if token_span else None)

    return extracted


def attn_logits_to_probs(attn: torch.Tensor) -> torch.Tensor:
    return torch.softmax(attn, dim=-1)


def update_head_stats(head_stats, layer_idx, t2i_attn, label):
    if layer_idx not in head_stats:
        num_heads = t2i_attn[0].shape[0]
        device = t2i_attn[0].device
        head_stats[layer_idx] = {
            "mass_sum": torch.zeros(num_heads, device=device),
            "alignment_score": torch.zeros(num_heads, device=device),
            "count": torch.zeros(num_heads, device=device),
        }

    mass_per_head = torch.stack([attn.sum(dim=(-1, -2)) for attn in t2i_attn if attn is not None])
    alignment_score_per_head = soft_suppression_loss(t2i_attn, label, per_head=True, temp=5)

    head_stats[layer_idx]["mass_sum"] += mass_per_head.mean(dim=0)
    head_stats[layer_idx]["alignment_score"] += alignment_score_per_head.sum(dim=0)
    head_stats[layer_idx]["count"] += alignment_score_per_head.shape[0]


def _elbow_threshold(values: torch.Tensor, prefer: str, threshold_pct: float = 0.7) -> tuple[float, int]:
    """Find threshold from a sorted curve using max distance to chord."""
    if values.numel() == 0:
        return float("nan"), -1
    if values.numel() == 1:
        return float(values[0].item()), 0

    if prefer not in {"low", "high"}:
        raise ValueError(f"Unsupported prefer mode: {prefer}")
    sorted_values, _ = torch.sort(values, descending=(prefer == "high"))

    x = torch.arange(sorted_values.numel(), dtype=torch.float32, device=sorted_values.device)
    y = sorted_values.float()

    p1 = torch.stack([x[0], y[0]])
    p2 = torch.stack([x[-1], y[-1]])
    line = p2 - p1
    line_norm = torch.linalg.norm(line)
    if line_norm.item() == 0.0:
        return float(sorted_values[0].item()), 0

    pts = torch.stack([x, y], dim=1)
    rel = pts - p1
    cross = torch.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0])
    distances = cross / line_norm

    max_dist, max_idx = torch.max(distances, dim=0)
    
    # Find the first index AFTER the max where distance drops to 80% of peak
    # This ensures you've fully rounded the corner.
    look_ahead = distances[max_idx:]
    threshold_mask = look_ahead < (max_dist * threshold_pct)
    
    if torch.any(threshold_mask):
        elbow_idx = max_idx.item() + int(torch.where(threshold_mask)[0][0].item())
    else:
        elbow_idx = int(max_idx.item())
    return float(sorted_values[elbow_idx].item()), elbow_idx


def select_grounding_heads(head_stats):
    candidates = [
        {
            "layer_idx": int(layer_idx),
            "head_idx": int(head_idx),
            "mass": float(mass.item()),
            "alignment": float(alignment.item()),
        }
        for layer_idx, stats in head_stats.items()
        for head_idx, (mass, alignment) in enumerate(
            zip(
                stats["mass_sum"] / stats["count"].clamp_min(1),
                stats["alignment_score"] / stats["count"].clamp_min(1),
            )
        )
    ]

    if not candidates:
        return {}, {"num_candidates": 0, "selected": 0}

    mass_values = torch.tensor([c["mass"] for c in candidates])
    alignment_values = torch.tensor([c["alignment"] for c in candidates])

    mass_thresh, mass_elbow_idx = _elbow_threshold(mass_values, prefer="high", threshold_pct=0.6)
    alignment_thresh, alignment_elbow_idx = _elbow_threshold(alignment_values, prefer="low", threshold_pct=1.0)

    selected = [c for c in candidates if c["mass"] >= mass_thresh and c["alignment"] <= alignment_thresh]

    selected_map = defaultdict(list)
    for c in selected:
        selected_map[c["layer_idx"]].append(c["head_idx"])
    selected_map = {layer_idx: sorted(heads) for layer_idx, heads in selected_map.items()}

    debug = {
        "num_candidates": len(candidates),
        "selected": len(selected),
        "mass_thresh": mass_thresh,
        "alignment_thresh": alignment_thresh,
        "mass_elbow_idx": mass_elbow_idx,
        "alignment_elbow_idx": alignment_elbow_idx,
        "selection_method": "elbow",
    }
    return selected_map, debug


class AttnHookManager:
    def __init__(self):
        self.attentions = {}
        self.hooks = []
        self.selected_heads_map = None

    def _hook_fn(self, layer_idx: int):
        """Internal hook to capture the second element of the layer output."""

        def hook(_module, _input, output):
            if isinstance(output, tuple) and len(output) == 3 and output[2] is not None:
                attn = output[2]
                if self.selected_heads_map is not None:
                    attn = attn[:, self.selected_heads_map.get(layer_idx, []), :, :]
                self.attentions[layer_idx] = attn

        return hook

    def attach(self, model: torch.nn.Module, selected_heads_map=None):
        """Registers hooks to all layers in the Qwen language model backbone."""
        self.remove_hooks()
        self.clear()  # Ensure no stale data
        self.selected_heads_map = selected_heads_map

        layers = model.model.model.language_model.layers  # qwen3-vl specific path to transformer layers
        for i, layer in enumerate(layers):
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
