import torch
from collections import defaultdict
from attn_ft.data import AttnBatch
from attn_ft.losses import soft_suppression_loss
import transformers


def _build_minicpm_token_targets(
    batch: AttnBatch,
    resampler_attn: torch.Tensor,
) -> list[torch.Tensor]:
    token_targets: list[torch.Tensor] = []
    visual_offset = 0

    for sample_idx, image_inputs in enumerate(batch.inputs["pixel_values"]):
        num_visual_inputs = len(image_inputs)
        expected_tokens = batch.image_token_indices[sample_idx].numel()
        sample_targets: list[torch.Tensor] = []

        for local_idx in range(num_visual_inputs):
            patch_mask = batch.vision_patch_masks[sample_idx][local_idx].to(
                device=resampler_attn.device,
                dtype=resampler_attn.dtype,
            )
            query_patch_attn = resampler_attn[visual_offset + local_idx, :, : patch_mask.numel()]
            sample_targets.append(query_patch_attn @ patch_mask)

        visual_offset += num_visual_inputs
        if sample_targets:
            token_target = torch.cat(sample_targets, dim=0)
        else:
            token_target = torch.empty(0, device=resampler_attn.device, dtype=resampler_attn.dtype)

        if token_target.numel() > expected_tokens:
            token_target = token_target[:expected_tokens]
        elif token_target.numel() < expected_tokens:
            token_target = torch.cat(
                [
                    token_target,
                    torch.zeros(expected_tokens - token_target.numel(), device=token_target.device, dtype=token_target.dtype),
                ],
                dim=0,
            )

        token_targets.append(token_target)

    return token_targets

def extract_t2i_attn_valid(
    attn: torch.Tensor,
    batch: AttnBatch,
    resampler_attn: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    extracted = []
    if batch.vision_patch_masks is not None:
        if resampler_attn is None:
            raise ValueError("MiniCPM supervision requires resampler attention weights")
        targets = _build_minicpm_token_targets(batch, resampler_attn)
    elif batch.vision_token_targets is not None:
        targets = batch.vision_token_targets
    else:
        targets = batch.masks

    masks = []
    for batch_idx in batch.valid_supervision_indices:
        token_span = batch.token_spans[batch_idx]
        if batch.vision_token_targets is not None:
            extracted.append(attn[batch_idx][:, token_span, : targets[batch_idx].numel()])
            masks.append(targets[batch_idx])
            continue

        image_idx = batch.image_token_indices[batch_idx]
        extracted.append(attn[batch_idx][:, token_span, image_idx])
        masks.append(targets[batch_idx])
    return extracted, masks


def update_head_stats(head_stats, layer_idx, t2i_attn, label):
    if not t2i_attn:
        return

    if layer_idx not in head_stats:
        num_heads = t2i_attn[0].shape[0]
        device = t2i_attn[0].device
        head_stats[layer_idx] = {
            "mass_sum": torch.zeros(num_heads, device=device),
            "alignment_score": torch.zeros(num_heads, device=device),
            "count": torch.zeros(num_heads, device=device),
        }

    t2i_attn_fp32 = [attn.float() for attn in t2i_attn]
    mass_per_head_list = []
    for attn in t2i_attn_fp32:
        finite_mask = torch.isfinite(attn)
        safe_attn = torch.where(finite_mask, attn, torch.zeros_like(attn))
        finite_count = finite_mask.sum(dim=(-1, -2)).clamp_min(1).to(attn.dtype)
        mass_per_head_list.append(safe_attn.sum(dim=(-1, -2)) / finite_count)
    mass_per_head = torch.stack(mass_per_head_list)
    alignment_score_per_head = soft_suppression_loss(t2i_attn_fp32, label, per_head=True, temp=5)

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


def select_grounding_heads(head_stats, allowed_layer_ids: list[int] | None = None):
    allowed_layer_set = set(allowed_layer_ids) if allowed_layer_ids is not None else None
    candidates = [
        {
            "layer_idx": int(layer_idx),
            "head_idx": int(head_idx),
            "mass": float(mass.item()),
            "alignment": float(alignment.item()),
        }
        for layer_idx, stats in head_stats.items()
        if allowed_layer_set is None or int(layer_idx) in allowed_layer_set
        for head_idx, (mass, alignment) in enumerate(
            zip(
                stats["mass_sum"] / stats["count"].clamp_min(1),
                stats["alignment_score"] / stats["count"].clamp_min(1),
            )
        )
        if torch.isfinite(mass) and torch.isfinite(alignment)
    ]

    if not candidates:
        return {}, {"num_candidates": 0, "selected": 0}

    mass_values = torch.tensor([c["mass"] for c in candidates])
    alignment_values = torch.tensor([c["alignment"] for c in candidates])

    mass_thresh, mass_elbow_idx = _elbow_threshold(mass_values, prefer="high", threshold_pct=1.0)
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
        self.resampler_attentions = []
        self.hooks = []
        self.selected_heads_map = None

    def _hook_fn(self, layer_idx: int):
        """Internal hook to capture the second element of the layer output."""

        def hook(_module, _input, output):
            attn = None
            if isinstance(output, tuple) and len(output) >= 3 and output[2] is not None:
                attn = output[2]
            elif isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn = output[1]

            if attn is None:
                return

            if self.selected_heads_map is not None:
                attn = attn[:, self.selected_heads_map.get(layer_idx, []), :, :]
            self.attentions[layer_idx] = attn

        return hook

    def _resampler_hook_fn(self):
        def hook(_module, _input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.resampler_attentions.append(output[1].detach())

        return hook

    def attach(self, model: torch.nn.Module, selected_heads_map=None):
        """Registers hooks to all layers in the Qwen language model backbone."""
        self.remove_hooks()
        self.clear()  # Ensure no stale data
        self.selected_heads_map = selected_heads_map
        candidates = [
            model,
            getattr(model, "base_model", None),
            getattr(getattr(model, "base_model", None), "model", None),
        ]
        core_model = next((candidate for candidate in candidates if candidate is not None), model)

        for candidate in candidates:
            if isinstance(candidate, transformers.Qwen3VLForConditionalGeneration):
                core_model = candidate
                break
            if candidate is not None and hasattr(candidate, "llm") and hasattr(candidate, "resampler"):
                core_model = candidate
                break
            if candidate is not None and hasattr(candidate, "language_model") and hasattr(candidate, "vision_model"):
                core_model = candidate
                break

        if isinstance(core_model, transformers.Qwen3VLForConditionalGeneration):
            layers = core_model.model.language_model.layers
        elif hasattr(core_model, "llm") and hasattr(core_model, "resampler"):
            layers = core_model.llm.model.layers
            handle = core_model.resampler.attn.register_forward_hook(self._resampler_hook_fn())
            self.hooks.append(handle)
        elif hasattr(core_model, "language_model") and hasattr(core_model, "vision_model"):
            layers = [
                core_model.language_model.layers[layer_idx].cross_attn
                for layer_idx in core_model.language_model.cross_attention_layers
            ]
        else:
            raise ValueError(f"Unsupported model type for AttnHookManager: {type(model)}")
        for i, layer in enumerate(layers):
            module = layer if hasattr(layer, "forward") and not hasattr(layer, "self_attn") else layer.self_attn
            handle = module.register_forward_hook(self._hook_fn(i))
            self.hooks.append(handle)

        print(f"Attached {len(self.hooks)} hooks to model layers.")

    def get_attentions(self) -> list[torch.Tensor]:
        """Returns a list of attention tensors ordered by layer index."""
        return [self.attentions[i] for i in sorted(self.attentions.keys())]

    def get_resampler_attention(self) -> torch.Tensor | None:
        if not self.resampler_attentions:
            return None
        if len(self.resampler_attentions) == 1:
            return self.resampler_attentions[0]
        return torch.cat(self.resampler_attentions, dim=0)

    def clear(self):
        """Clears stored attention tensors."""
        self.attentions.clear()
        self.resampler_attentions.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
