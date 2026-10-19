"""WT protocol and full-model regularization ablations."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path

DATA_ROOT = os.environ.get("WT_DATA_ROOT", str(Path("data").resolve()))
CLASSES = ["healthy", "broken", "missing_tooth", "root_crack", "wear"]
NUM_CLASSES = len(CLASSES)
HEALTHY_CLASS = "healthy"
SEG_LEN = 4096
ANGLE_N_PER_REV = 1024
ANGLE_REVS_PER_SEG = 4
DEFAULT_TRAIN_RATIO = 0.85
DEFAULT_MAX_SEGMENTS_PER_FILE = 160
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42
DEFAULT_SEG_DOMAIN = "time"
ENCODER_NORM = "batch"
INPUT_NORMALIZATION = "healthy"
SEEDS = tuple(range(42, 52))
VARIANTS = ("full", "no_mi", "no_diversity", "no_both", "no_alignment")
MAPPINGS = ("linear", "mlp", "fixed", "all_class")


def class_to_idx():
    return {name: i for i, name in enumerate(CLASSES)}


@dataclass
class Experiment:
    data_root: str = DATA_ROOT
    reference_protocol: str = "disjoint20_random3_v1"
    variant: str = "full"
    mapping: str = "linear"
    model_type: str = "imacd"
    installation: int = 1
    seed: int = 42
    encoder_norm: str = "batch"
    encoder_sharing: str = "independent"
    input_normalization: str = "healthy"
    embed_dim: int = 256
    backbone_channels: int = 64
    epochs: int = 10
    batch_size: int = 64
    num_workers: int = 0
    max_segments_per_file: int = 160
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    temperature: float = 0.2
    mine_hidden: int = 256
    device: str = "cuda"

    def __post_init__(self):
        self.data_root = str(self.data_root)
        if self.reference_protocol != "disjoint20_random3_v1":
            raise ValueError("Unsupported reference protocol; legacy checkpoints require their archived loaders")
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")
        if self.mapping not in MAPPINGS:
            raise ValueError(f"Unknown mapping: {self.mapping}")
        if self.model_type not in ("imacd", "query", "concat", "siamese"):
            raise ValueError("Unknown model_type")
        if self.model_type != "imacd" and (self.variant != "full" or self.mapping != "linear"):
            raise ValueError("Mapping and objective variants apply only to Ima-CD")
        if self.installation not in (1, 2):
            raise ValueError("installation must be 1 or 2")
        if self.encoder_norm not in ("batch", "group"):
            raise ValueError("encoder_norm must be batch or group")
        if self.encoder_sharing not in ("independent", "shared"):
            raise ValueError("encoder_sharing must be independent or shared")
        if self.input_normalization not in ("healthy", "window"):
            raise ValueError("input_normalization must be healthy or window")
        if self.encoder_norm == "group" and self.backbone_channels % 8:
            raise ValueError("GroupNorm requires backbone_channels divisible by 8")
        if min(self.epochs, self.batch_size, self.max_segments_per_file,
               self.embed_dim, self.backbone_channels, self.mine_hidden) <= 0 or self.num_workers < 0:
            raise ValueError("Model dimensions and training budgets must be positive")
        if not (0 < self.min_lr <= self.lr and self.temperature > 0
                and self.weight_decay >= 0 and self.grad_clip > 0):
            raise ValueError("Invalid optimizer or temperature setting")
        if self.num_workers != 0:
            raise ValueError("The recorded episode sampling protocol requires num_workers=0")

    @property
    def alignment_weight(self):
        return 0.5 if self.model_type == "imacd" and self.variant != "no_alignment" else 0.0

    @property
    def mi_weight(self):
        return 0.02 if self.model_type == "imacd" and self.variant not in ("no_mi", "no_both") else 0.0

    @property
    def diversity_weight(self):
        return 5.0 if self.model_type == "imacd" and self.variant not in ("no_diversity", "no_both") else 0.0

    @property
    def name(self):
        if self.model_type != "imacd":
            return (f"{self.model_type}_{self.encoder_norm}_{self.input_normalization}_"
                    f"i{self.installation}_seed{self.seed}")
        return (f"{self.mapping}_{self.variant}_ALIGN{self.alignment_weight:g}_"
                f"MI{self.mi_weight:g}_DIV{self.diversity_weight:g}_{self.encoder_norm}_"
                f"{self.input_normalization}_{self.encoder_sharing}_"
                f"i{self.installation}_seed{self.seed}")


@contextmanager
def data_settings(data_root, normalization="healthy"):
    global DATA_ROOT, INPUT_NORMALIZATION
    previous = DATA_ROOT, INPUT_NORMALIZATION
    DATA_ROOT, INPUT_NORMALIZATION = str(data_root), normalization
    try:
        yield
    finally:
        DATA_ROOT, INPUT_NORMALIZATION = previous
