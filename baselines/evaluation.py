"""Baseline inference reconstructed from the checkpoint's own settings."""
import numpy as np
import torch

from imacd.utils import confusion_matrix, load_checkpoint, macro_f1, set_seed
from . import config as CT
from .data import build_plain_loader
from .models import build_model


@torch.no_grad()
def evaluate(checkpoint, *, data_root=None, device="cpu"):
    state = load_checkpoint(checkpoint, device)
    cfg = state["cfg"]
    if cfg.get("reference_protocol") != "disjoint20_random3_v1":
        raise ValueError("Legacy checkpoint uses the former query set; use its archived evaluation protocol")
    set_seed(cfg["seed"])
    with CT.data_settings(data_root if data_root is not None else cfg["data_root"]):
        loader, _ = build_plain_loader(
            cfg["target_installations"], cfg["target_speeds"],
            batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            max_segments_per_file=cfg["max_segments_per_file"], seg_domain=state["seg_domain"],
            seed=cfg["seed"])
        model = build_model(state["method"], CT.NUM_CLASSES, cfg["embed_dim"],
                            cfg["backbone_channels"], cfg["disc_hidden"],
                            state["n_domains"], state["target_mode"]).to(device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        truth, predictions = [], []
        for x, labels, _ in loader:
            truth.extend(labels.tolist())
            predictions.extend(model(x.to(device))["logits"].argmax(1).cpu().tolist())
        if not truth:
            raise ValueError("Cannot evaluate an empty loader")
        truth, predictions = np.asarray(truth), np.asarray(predictions)
        cm = confusion_matrix(truth, predictions, CT.NUM_CLASSES)
        return dict(accuracy=float((truth == predictions).mean()), macro_f1=macro_f1(cm),
                    confusion=cm.tolist(), n=len(truth),
                    labels=truth.tolist(), predictions=predictions.tolist())
