"""Query-only and healthy-reference fusion controls for the paired protocol."""
import torch
import torch.nn.functional as F
from torch import nn

from imacd.backbone import ConvEncoder1D
from imacd.model import mean_normal_embedding


class PairedClassifier(nn.Module):
    def __init__(self, method, num_classes=5, embed_dim=256, channels=64, encoder_norm="batch"):
        super().__init__()
        if method not in ("query", "concat", "siamese"):
            raise ValueError(f"Unknown paired baseline: {method}")
        self.method = method
        self.encoder = ConvEncoder1D(embed_dim, channels, encoder_norm)
        if method == "query":
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            width = embed_dim * 2 if method == "concat" else embed_dim
            self.classifier = nn.Sequential(nn.Linear(width, embed_dim), nn.ReLU(inplace=True),
                                            nn.Linear(embed_dim, num_classes))

    def forward(self, normal, query):
        if self.method == "query":
            features = F.normalize(self.encoder(query), dim=-1)
        else:
            reference = F.normalize(mean_normal_embedding(self.encoder, normal), dim=-1)
            query = F.normalize(self.encoder(query), dim=-1)
            features = (torch.cat([reference, query], dim=-1) if self.method == "concat"
                        else (reference - query).abs())
        return {"logits": self.classifier(features)}
