"""Adaptation losses for the transfer-learning baselines."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ramp_lambda(progress: float, gamma: float = 10.0) -> float:
    """DANN's 2/(1+exp(-gamma*p)) - 1 warm-up over training progress p in [0,1].

    Alignment terms start near zero and reach their full weight late, which
    stops the discriminator from tearing apart features that are still random.
    """
    p = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def _covariance(x: torch.Tensor) -> torch.Tensor:
    """Batch feature covariance, (D, D)."""
    n = x.size(0)
    xm = x - x.mean(dim=0, keepdim=True)
    return xm.t() @ xm / max(n - 1, 1)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """||C_s - C_t||_F^2 / (4 d^2), the Deep CORAL penalty (Sun & Saenko 2016).

    Both batches must be at least 2 samples; with fewer the covariance is
    undefined and the term is skipped (returns 0).
    """
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_zeros(())
    d = source.size(1)
    diff = _covariance(source) - _covariance(target)
    return (diff * diff).sum() / (4.0 * d * d)


def pairwise_coral_loss(feat: torch.Tensor, domain_index: torch.Tensor) -> torch.Tensor:
    """Mean CORAL over every pair of sub-domains inside one batch.

    Used in domain-generalization mode, where there is no target batch to
    align against, so the source speeds are aligned against each other.
    """
    doms = torch.unique(domain_index)
    if doms.numel() < 2:
        return feat.new_zeros(())
    parts = [feat[domain_index == d] for d in doms]
    terms = [coral_loss(parts[i], parts[j])
             for i in range(len(parts)) for j in range(i + 1, len(parts))]
    if not terms:
        return feat.new_zeros(())
    return torch.stack(terms).mean()


def binary_domain_loss(logits: torch.Tensor, is_target: bool) -> torch.Tensor:
    """BCE against an all-source (0) or all-target (1) label vector."""
    target = torch.full_like(logits, 1.0 if is_target else 0.0)
    return F.binary_cross_entropy_with_logits(logits, target)


def classifier_discrepancy(logits_stack: torch.Tensor) -> torch.Tensor:
    """Mean L1 distance between every pair of classifiers' softmax outputs.

    logits_stack: (K, B, C). Drives the K MSDAL heads to agree on the target
    batch, which is the condition under which their average is meaningful.
    """
    k = logits_stack.size(0)
    if k < 2:
        return logits_stack.new_zeros(())
    probs = logits_stack.softmax(dim=-1)
    terms = [(probs[i] - probs[j]).abs().mean()
             for i in range(k) for j in range(i + 1, k)]
    return torch.stack(terms).mean()


def softmax_aggregate(losses: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """MDAN's soft-max over per-source losses: (1/gamma) log sum exp(gamma*L_k).

    A smooth maximum, so optimization concentrates on the worst source rather
    than letting easy sources dilute the objective. Reduces to the max as
    gamma grows and to log(K)/gamma + mean as gamma shrinks.
    """
    return torch.logsumexp(gamma * losses, dim=0) / gamma
