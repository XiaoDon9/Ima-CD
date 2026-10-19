"""Source-only, Deep CORAL, DANN and a local multi-source adversarial baseline.

Classification and alignment both use L2-normalized encoder features."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from imacd.backbone import ConvEncoder1D


class _GradReverse(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward (Ganin et al.)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """MLP domain head. n_out=1 -> binary (source vs target) with BCE logits;
    n_out=K -> K-way domain classification with cross-entropy."""

    def __init__(self, embed_dim: int, hidden: int = 256, n_out: int = 1):
        super().__init__()
        self.n_out = n_out
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EncoderClassifier(nn.Module):
    """Shared skeleton: encoder + linear head on the normalized embedding."""

    def __init__(self, num_classes: int, embed_dim: int, channels: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.encoder = ConvEncoder1D(embed_dim=embed_dim, channels=channels, encoder_norm="batch")
        self.classifier = nn.Linear(embed_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized embedding: the input to BOTH the classifier and the
        alignment objectives (see the module docstring)."""
        return F.normalize(self.encoder(x), dim=-1)

    def classify(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(feat)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        return {"feat": feat, "logits": self.classify(feat)}


class SourceOnly(_EncoderClassifier):
    """No adaptation: the lower bound every UDA number is read against."""


class DeepCORAL(_EncoderClassifier):
    """Encoder/classifier trained with a source-target covariance penalty."""


class DANN(_EncoderClassifier):
    """Gradient-reversal domain classification: binary UDA or source-domain DG."""

    def __init__(self, num_classes: int, embed_dim: int, channels: int = 64,
                 disc_hidden: int = 256, n_domains: int = 2, binary: bool = True):
        super().__init__(num_classes, embed_dim, channels)
        self.binary = binary
        self.discriminator = DomainDiscriminator(
            embed_dim, hidden=disc_hidden, n_out=1 if binary else n_domains
        )

    def domain_logits(self, feat: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.discriminator(grad_reverse(feat, lambd))


class MSDAL(_EncoderClassifier):
    """One classifier and binary discriminator per source domain.
    
    Prediction averages class probabilities; training also penalizes classifier disagreement."""

    def __init__(self, num_classes: int, embed_dim: int, channels: int = 64,
                 disc_hidden: int = 256, n_domains: int = 4):
        super().__init__(num_classes, embed_dim, channels)
        self.n_domains = n_domains
        # The inherited `classifier` is unused; the per-source heads replace
        # it. It is deleted so it cannot silently absorb gradients or bloat
        # the checkpoint.
        del self.classifier
        self.classifiers = nn.ModuleList(
            nn.Linear(embed_dim, num_classes) for _ in range(n_domains)
        )
        self.discriminators = nn.ModuleList(
            DomainDiscriminator(embed_dim, hidden=disc_hidden, n_out=1)
            for _ in range(n_domains)
        )

    def classify_with(self, feat: torch.Tensor, k: int) -> torch.Tensor:
        """Logits from source k's head."""
        return self.classifiers[k](feat)

    def all_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """(K, B, C) logits, every head applied to every sample."""
        return torch.stack([head(feat) for head in self.classifiers], dim=0)

    def classify(self, feat: torch.Tensor) -> torch.Tensor:
        # log of the mean softmax: a proper ensemble score whose argmax is the
        # ensemble prediction.
        return torch.log(self.all_logits(feat).softmax(dim=-1).mean(dim=0) + 1e-12)

    def domain_logits(self, feat: torch.Tensor, k: int, lambd: float) -> torch.Tensor:
        return self.discriminators[k](grad_reverse(feat, lambd))


def build_model(
    method: str,
    num_classes: int,
    embed_dim: int,
    channels: int = 64,
    disc_hidden: int = 256,
    n_domains: int = 1,
    target_mode: str = "unlabeled",
) -> nn.Module:
    method = (method or "source_only").lower()
    if method == "source_only":
        return SourceOnly(num_classes, embed_dim, channels)
    if method == "coral":
        return DeepCORAL(num_classes, embed_dim, channels)
    if method == "dann":
        # UDA: binary source-vs-target. DG: K-way over the source sub-domains.
        binary = target_mode == "unlabeled"
        return DANN(num_classes, embed_dim, channels, disc_hidden=disc_hidden,
                    n_domains=n_domains, binary=binary)
    if method == "msdal":
        return MSDAL(num_classes, embed_dim, channels, disc_hidden=disc_hidden,
                     n_domains=n_domains)
    raise ValueError(f"unknown method={method!r}; choose from "
                     f"('source_only', 'dann', 'coral', 'msdal')")
