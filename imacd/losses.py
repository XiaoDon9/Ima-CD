"""Classification, healthy alignment, DV-style MI and map diversity."""
import torch
import torch.nn.functional as F
from torch import nn


class MINE(nn.Module):
    """Finite-batch DV-style dependency critic used only during training."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(inplace=True),
                               nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                               nn.Linear(hidden, 1))

    def forward(self, query, anchor):
        joint = self.t(torch.cat([query, anchor], dim=-1))
        shuffled = anchor[torch.randperm(anchor.size(0), device=anchor.device)]
        marginal = self.t(torch.cat([query, shuffled], dim=-1))
        bound = joint.mean() - torch.log(torch.exp(marginal.clamp(-20, 20)).mean() + 1e-8)
        return -bound


def map_diversity(weights, margin=2 ** 0.5):
    if weights.size(0) < 2:
        return weights.new_zeros(())
    directions = F.normalize(weights.reshape(weights.size(0), -1), dim=-1)
    pairs = torch.triu_indices(weights.size(0), weights.size(0), offset=1, device=weights.device)
    difference = directions[pairs[0]] - directions[pairs[1]]
    distance = ((difference * difference).sum(-1) + 1e-12).sqrt()
    return F.relu(margin - distance).mean()


def healthy_alignment(z_q, z_n, labels):
    healthy = labels == 0
    if not healthy.any():
        return z_q.new_zeros(())
    return (1 - F.cosine_similarity(z_q[healthy], z_n[healthy], dim=-1)).mean()


def objective(output, labels, weights, critic=None, *, temperature=0.2,
              alignment_weight=0.5, mi_weight=0.02, diversity_weight=5.0):
    if mi_weight and critic is None:
        raise ValueError("MI regularization requires a critic")
    # Single-term ablations retain the critic and its permutation, matching
    # the original training trajectory; joint removal omits the critic.
    mi = (critic(output["z_q"], output["anchors"][torch.arange(labels.size(0), device=labels.device), labels])
          if critic is not None else output["z_q"].new_zeros(()))
    diversity = map_diversity(weights) if diversity_weight else output["z_q"].new_zeros(())
    alignment = (healthy_alignment(output["z_q"], output["z_n"], labels)
                 if alignment_weight else output["z_q"].new_zeros(()))
    classification = F.cross_entropy(output["logits"] / temperature, labels)
    loss = mi_weight * mi + classification if critic is not None else classification
    if diversity_weight:
        loss = loss + diversity_weight * diversity
    if alignment_weight:
        loss = loss + alignment_weight * alignment
    return loss, dict(classification=classification, alignment=alignment,
                      mi=mi, diversity=diversity)
