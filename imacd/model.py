"""Healthy-anchored fault-feature imagination."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import ConvEncoder1D
from .config import MAPPINGS


def mean_normal_embedding(encoder, normal):
    batch, refs, length = normal.shape
    features = encoder(normal.reshape(batch * refs, 1, length))
    return features.reshape(batch, refs, -1).mean(dim=1)


class ImaCD(nn.Module):
    """Linear/MLP imagination and explicit candidate-bank controls."""

    def __init__(self, num_classes=5, embed_dim=256, channels=64,
                 encoder_norm="batch", encoder_sharing="independent",
                 mapping="linear", retain_fixed_reference=True):
        super().__init__()
        if encoder_sharing not in ("shared", "independent"):
            raise ValueError("Invalid encoder sharing")
        if num_classes < 2:
            raise ValueError("At least healthy and one fault class are required")
        if mapping not in MAPPINGS:
            raise ValueError(f"Unknown mapping: {mapping}")
        self.num_classes = num_classes
        self.mapping = mapping
        if mapping == "fixed":
            self.fixed_seed = nn.Parameter(torch.randn(embed_dim))
        if mapping == "fixed" and not retain_fixed_reference:
            self.query_encoder = ConvEncoder1D(embed_dim, channels, encoder_norm)
        else:
            self.normal_encoder = ConvEncoder1D(embed_dim, channels, encoder_norm)
            self.query_encoder = (self.normal_encoder if encoder_sharing == "shared"
                                  else ConvEncoder1D(embed_dim, channels, encoder_norm))
        if mapping == "mlp":
            self.fault_maps = nn.ModuleList([
                nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                              nn.Linear(embed_dim, embed_dim)) for _ in range(num_classes - 1)])
            for transform in self.fault_maps:
                for layer in transform:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight)
                        nn.init.zeros_(layer.bias)
        else:
            n_maps = num_classes if mapping == "all_class" else num_classes - 1
            weights = torch.empty(n_maps, embed_dim, embed_dim)
            for weight in weights:
                nn.init.orthogonal_(weight)
            self.label_weights = nn.Parameter(weights)

    def diversity_parameters(self):
        if self.mapping == "mlp":
            return torch.stack([torch.cat([p.reshape(-1) for p in transform.parameters()])
                                for transform in self.fault_maps])
        return self.label_weights

    def forward(self, normal, query):
        if self.mapping == "fixed":
            candidate_input = F.normalize(self.fixed_seed, dim=-1).expand(query.size(0), -1)
            z_n = (F.normalize(mean_normal_embedding(self.normal_encoder, normal), dim=-1)
                   if hasattr(self, "normal_encoder") else candidate_input)
        else:
            z_n = F.normalize(mean_normal_embedding(self.normal_encoder, normal), dim=-1)
            candidate_input = z_n
        z_q = F.normalize(self.query_encoder(query), dim=-1)
        mapped = (torch.stack([transform(candidate_input) for transform in self.fault_maps], dim=1)
                  if self.mapping == "mlp" else
                  torch.einsum("bd,cde->bce", candidate_input, self.label_weights))
        anchors = (mapped if self.mapping == "all_class" else
                   torch.cat([candidate_input.unsqueeze(1), mapped], dim=1))
        query_features = z_q.unsqueeze(1).expand(-1, anchors.size(1), -1)
        return dict(z_n=z_n, z_q=z_q, anchors=anchors,
                    logits=F.cosine_similarity(anchors, query_features, dim=-1))
