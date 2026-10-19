"""Training/evaluation APIs; no sweep launchers or archived artifacts."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .config import Experiment, NUM_CLASSES, data_settings
from .data import build_test_loader, build_train_val_loaders
from .losses import MINE, objective
from .model import ImaCD
from .utils import confusion_matrix, load_checkpoint, macro_f1, save_checkpoint, set_seed


def build_model(cfg):
    if cfg.model_type != "imacd":
        from baselines.paired import PairedClassifier
        return PairedClassifier(cfg.model_type, NUM_CLASSES, cfg.embed_dim,
                                cfg.backbone_channels, cfg.encoder_norm)
    return ImaCD(NUM_CLASSES, cfg.embed_dim, cfg.backbone_channels,
                 cfg.encoder_norm, cfg.encoder_sharing, cfg.mapping,
                 retain_fixed_reference=bool(cfg.alignment_weight))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    truth, predicted = [], []
    for normal, query, labels in loader:
        scores = model(normal.to(device), query.to(device))["logits"]
        truth.extend(labels.tolist())
        predicted.extend(scores.argmax(1).cpu().tolist())
    if not truth:
        raise ValueError("Cannot evaluate an empty loader")
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    cm = confusion_matrix(truth, predicted, NUM_CLASSES)
    return dict(accuracy=float((truth == predicted).mean()), macro_f1=macro_f1(cm),
                confusion=cm.tolist(), n=len(truth),
                labels=truth.tolist(), predictions=predicted.tolist())


def train(cfg: Experiment, output_dir):
    """Train one configuration; existing run directories are never overwritten."""
    cfg = Experiment(**asdict(cfg))
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device='cpu' explicitly")
    path = Path(output_dir) / cfg.name
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    set_seed(cfg.seed)
    with data_settings(cfg.data_root, cfg.input_normalization):
        train_loader, val_loader = build_train_val_loaders(
            [cfg.installation], [20, 25, 30, 35], batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, max_segments_per_file=cfg.max_segments_per_file,
            seed=cfg.seed, seg_domain="time",
            val_installations=[cfg.installation], val_speeds=[40])
        if not len(train_loader) or not len(val_loader):
            raise ValueError("Training and validation loaders must be nonempty")
        model = build_model(cfg).to(device)
        critic = (MINE(cfg.embed_dim, cfg.mine_hidden).to(device)
                  if cfg.model_type == "imacd" and cfg.variant != "no_both" else None)
        parameters = list(model.parameters()) + (list(critic.parameters()) if critic is not None else [])
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
        path.mkdir(parents=True)
        writer = SummaryWriter(str(path / "tb"))
        writer.add_text("configuration", json.dumps(asdict(cfg), indent=2), 0)
        history, best, best_epoch, step = [], -1.0, None, 0
        try:
            for epoch in range(1, cfg.epochs + 1):
                model.train()
                if critic is not None:
                    critic.train()
                lr = optimizer.param_groups[0]["lr"]
                totals = dict.fromkeys(("loss", "classification", "alignment", "mi", "diversity"), 0.0)
                for normal, query, labels in train_loader:
                    output = model(normal.to(device), query.to(device))
                    if cfg.model_type == "imacd":
                        loss, terms = objective(
                            output, labels.to(device), model.diversity_parameters(), critic,
                            temperature=cfg.temperature, alignment_weight=cfg.alignment_weight,
                            mi_weight=cfg.mi_weight, diversity_weight=cfg.diversity_weight)
                    else:
                        classification = F.cross_entropy(output["logits"], labels.to(device))
                        loss = classification
                        terms = dict(classification=classification,
                                     alignment=classification.new_zeros(()),
                                     mi=classification.new_zeros(()), diversity=classification.new_zeros(()))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
                    optimizer.step()
                    for key, value in dict(loss=loss, **terms).items():
                        scalar = float(value.detach())
                        totals[key] += scalar
                        writer.add_scalar("train/" + key, scalar, step)
                    step += 1
                scheduler.step()
                val = predict(model, val_loader, device)
                row = dict(epoch=epoch, lr=lr,
                           **{key: value / len(train_loader) for key, value in totals.items()},
                           val_accuracy=val["accuracy"], batches=len(train_loader))
                history.append(row)
                for key in ("lr", "loss", "classification", "alignment", "mi", "diversity", "val_accuracy"):
                    writer.add_scalar("epoch/" + key, row[key], epoch)
                state = dict(model_state_dict=model.state_dict(), cfg=asdict(cfg), epoch=epoch,
                             val_acc=val["accuracy"], optimizer_state_dict=optimizer.state_dict(),
                             scheduler_state_dict=scheduler.state_dict())
                if critic is not None:
                    state["mine_state_dict"] = critic.state_dict()
                if val["accuracy"] >= best:
                    best, best_epoch = val["accuracy"], epoch
                    save_checkpoint(state, path / "best.pt")
                if epoch == cfg.epochs:
                    save_checkpoint(state, path / "last.pt")
            state = load_checkpoint(path / "best.pt", device)
            model.load_state_dict(state["model_state_dict"], strict=True)
            set_seed(cfg.seed)
            loader = build_test_loader(
                [cfg.installation], [45, 50, 55], batch_size=cfg.batch_size,
                num_workers=cfg.num_workers, max_segments_per_file=cfg.max_segments_per_file,
                seed=cfg.seed, seg_domain="time")
            metrics = predict(model, loader, device)
            result = dict(cfg=asdict(cfg), best_epoch=best_epoch, best_val_accuracy=best,
                          history=history, test=metrics)
            (path / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            writer.add_scalar("test/accuracy", metrics["accuracy"], best_epoch)
            writer.add_scalar("test/macro_f1", metrics["macro_f1"], best_epoch)
            return result
        finally:
            writer.close()


def evaluate(checkpoint, *, data_root=None, device="cpu"):
    """Restore architecture, normalization, installation and seed from a checkpoint."""
    state = load_checkpoint(checkpoint, device)
    if state["cfg"].get("reference_protocol") != "disjoint20_random3_v1":
        raise ValueError("Legacy checkpoint uses overlapping references; evaluate with its archived protocol")
    cfg = Experiment(**state["cfg"])
    if data_root is not None:
        cfg.data_root = str(data_root)
    set_seed(cfg.seed)
    with data_settings(cfg.data_root, cfg.input_normalization):
        model = build_model(cfg).to(device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        loader = build_test_loader(
            [cfg.installation], [45, 50, 55], batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, max_segments_per_file=cfg.max_segments_per_file,
            seed=cfg.seed, seg_domain="time")
        return predict(model, loader, torch.device(device))
