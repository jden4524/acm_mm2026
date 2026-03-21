from __future__ import annotations
import torch


def _zero_like(pred: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
    for tensor in pred:
        if tensor is not None and tensor.numel() > 0:
            return tensor.new_zeros(())
    for tensor in target:
        if tensor is not None and tensor.numel() > 0:
            return tensor.new_zeros(())
    return torch.tensor(0.0)


def _prepare_padded_inputs(
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not pred or not target:
        return None

    zero = _zero_like(pred, target)
    num_heads: int | None = None
    valid: list[tuple[torch.Tensor, torch.Tensor]] = []
    max_t = 0
    max_s = 0

    for p, t in zip(pred, target):
        if p is None or t is None or p.numel() == 0 or t.numel() == 0:
            continue
        if p.dim() != 3:
            continue

        if num_heads is None:
            num_heads = p.shape[0]
        if p.shape[0] != num_heads:
            continue

        t_flat = t.to(device=p.device, dtype=p.dtype).reshape(-1)
        if p.shape[-1] != t_flat.shape[0]:
            continue

        valid.append((p, t_flat))
        max_t = max(max_t, p.shape[1])
        max_s = max(max_s, p.shape[2])

    if not valid or num_heads is None:
        return None

    n = len(valid)
    device = valid[0][0].device
    dtype = valid[0][0].dtype

    logits = torch.zeros((n, num_heads, max_t, max_s), device=device, dtype=dtype)
    targets = torch.zeros((n, max_s), device=device, dtype=dtype)
    text_mask = torch.zeros((n, max_t), device=device, dtype=torch.bool)
    image_mask = torch.zeros((n, max_s), device=device, dtype=torch.bool)

    for i, (p, t_flat) in enumerate(valid):
        t_len = p.shape[1]
        s_len = p.shape[2]
        logits[i, :, :t_len, :s_len] = p
        targets[i, :s_len] = t_flat
        text_mask[i, :t_len] = True
        image_mask[i, :s_len] = True

    return logits, targets, text_mask, image_mask, zero


def soft_suppression_loss(
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    per_head: bool = False,
    temp: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    packed = _prepare_padded_inputs(pred, target)
    if packed is None:
        return _zero_like(pred, target)

    logits, targets, text_mask, image_mask, zero = packed
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")

    valid_image = image_mask[:, None, None, :]
    target_4d = targets[:, None, None, :]
    neg_inf = torch.finfo(logits.dtype).min

    pos_logits = (logits * target_4d).masked_fill(~valid_image, neg_inf)
    neg_logits = (logits * (1.0 - target_4d) / temp).masked_fill(~valid_image, neg_inf)

    log_sum_pos = torch.logsumexp(pos_logits, dim=-1)
    log_sum_neg = torch.logsumexp(neg_logits, dim=-1)
    loss_per_text = torch.logaddexp(log_sum_pos, log_sum_neg) - log_sum_pos

    valid_text = text_mask[:, None, :]
    masked_loss = loss_per_text * valid_text
    text_counts = text_mask.sum(dim=-1).clamp_min(1).to(logits.dtype)

    if per_head:
        return masked_loss.sum(dim=-1) / text_counts[:, None]

    num_heads = logits.shape[1]
    if num_heads == 0:
        return zero

    sample_loss = masked_loss.sum(dim=(-1, -2)) / (text_counts * num_heads)
    return sample_loss.sum()
