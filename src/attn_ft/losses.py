from __future__ import annotations
import torch.nn.functional as F
import torch


def ce_loss(pred: list[torch.Tensor], target: list[torch.Tensor], eps: float = 1.0e-6) -> torch.Tensor:
    losses = []
    if len(pred) == 0 or len(target) == 0:
        return 0
    
    for p, t in zip(pred, target):
        if t is not None and p is not None:
            
            t = t.to(p.device).view(-1)
  
            rescaled_p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
            a = rescaled_p * t
            a_agg = a.mean(dim=-2)

            loss = -(t*torch.log(a_agg + 1e-8)).sum(dim=-1).mean()
        
            losses.append(loss)
    
    if len(losses) == 0:
        return 0
    return torch.stack(losses).mean()


def vacuum_loss(pred: list[torch.Tensor], target: list[torch.Tensor], tau=1.0):
    losses = []
    if len(pred) == 0 or len(target) == 0:
        return 0
    
    for p, t in zip(pred, target):
        if t is not None and p is not None:
            
            t = t.to(p.device).view(-1)
            
            rescaled_p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
            a = rescaled_p * t
            a_agg = a.mean(dim=-2)

            loss = -torch.log((a_agg*t).sum(dim=-1) + 1e-8).mean()
        
            losses.append(loss)
    
    if len(losses) == 0:
        return 0
    return torch.stack(losses).mean()


def soft_suppression_loss(pred: list[torch.Tensor], target: list[torch.Tensor]):
    losses = []
    if len(pred) == 0 or len(target) == 0:
        return 0
    
    for p, t in zip(pred, target):
        if t is not None and p is not None:
            
            t = t.to(p.device).view(-1)
            
            log_sum_pos = torch.logsumexp(p*t, dim=-1)
            log_sum_neg = torch.logsumexp(p*(1-t), dim=-1)

            loss = torch.logaddexp(log_sum_pos, log_sum_neg) - log_sum_pos
            loss = loss.mean()
        
            losses.append(loss)
    
    if len(losses) == 0:
        return 0
    return torch.stack(losses).mean()
