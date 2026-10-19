"""Training entry for the transfer-learning baselines on the WT gearbox."""
from __future__ import annotations

import os
import types

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from . import config as CT
from imacd.utils import save_checkpoint, set_seed
from .data import (build_plain_loader, build_source_loader, fmt_int_list, infinite,
                     parse_int_list)
from .losses import (binary_domain_loss, classifier_discrepancy, coral_loss,
                       pairwise_coral_loss, ramp_lambda, softmax_aggregate)
from .models import build_model


@torch.no_grad()
def validate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)["logits"].argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def _dann_step(model, feat_s, d_s, feat_t, lambd, binary):
    """Domain loss for DANN. UDA: source vs target. DG: K-way over sources."""
    if binary:
        src = binary_domain_loss(model.domain_logits(feat_s, lambd), is_target=False)
        tgt = binary_domain_loss(model.domain_logits(feat_t, lambd), is_target=True)
        return 0.5 * (src + tgt)
    return F.cross_entropy(model.domain_logits(feat_s, lambd), d_s)


def _msdal_step(model, feat_s, y_s, d_s, feat_t, lambd, cfg):
    """Per-source classification + adversarial losses, then aggregation.

    Returns (aggregated_loss, mean_cls_loss, mean_adv_loss, discrepancy).
    """
    uda = feat_t is not None
    per_source = []
    cls_terms = []
    adv_terms = []
    for k in range(model.n_domains):
        mask = d_s == k
        if not bool(mask.any()):
            continue
        feat_k = feat_s[mask]
        cls_k = F.cross_entropy(model.classify_with(feat_k, k), y_s[mask])

        # Discriminator k separates its own source from the target (UDA) or
        # from the pooled remaining sources (DG).
        other = feat_t if uda else feat_s[~mask]
        if other is not None and other.size(0) > 0:
            own = binary_domain_loss(model.domain_logits(feat_k, k, lambd), is_target=False)
            rest = binary_domain_loss(model.domain_logits(other, k, lambd), is_target=True)
            adv_k = 0.5 * (own + rest)
        else:
            adv_k = feat_s.new_zeros(())

        per_source.append(cls_k + cfg.lambda_msdal_adv * adv_k)
        cls_terms.append(cls_k)
        adv_terms.append(adv_k)

    stacked = torch.stack(per_source)
    if cfg.msdal_agg == "softmax":
        agg = softmax_aggregate(stacked, gamma=cfg.msdal_gamma)
    else:
        agg = stacked.mean()

    # Agreement between the K heads, on the target if we have it.
    disc_feat = feat_t if uda else feat_s
    disc = classifier_discrepancy(model.all_logits(disc_feat))
    return agg, torch.stack(cls_terms).mean(), torch.stack(adv_terms).mean(), disc


def train(cfg: types.SimpleNamespace, output_dir) -> str:
    """Train one baseline using the recorded baseline optimization protocol."""
    cfg = types.SimpleNamespace(**vars(cfg))
    if cfg.method not in CT.METHODS or cfg.target_mode not in CT.TARGET_MODES:
        raise ValueError("Unknown baseline method or target mode")
    if cfg.msdal_agg not in ("mean", "softmax"):
        raise ValueError("msdal_agg must be mean or softmax")
    if (cfg.epochs <= 0 or cfg.batch_size <= 0 or cfg.lr <= 0
            or cfg.grad_clip <= 0 or cfg.max_segments_per_file <= 0):
        raise ValueError("Training budgets and optimizer settings must be positive")
    device = torch.device("cuda" if cfg.device == "gpu" else cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device='cpu' explicitly")
    run_name = (f"{cfg.method}_{cfg.target_mode}_i"
                f"{fmt_int_list(parse_int_list(cfg.train_installations, [1]))}_seed{cfg.seed}")
    run_dir = os.path.join(output_dir, run_name)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    with CT.data_settings(cfg.data_root):
        return _train(cfg, device, run_dir)


def _train(cfg, device, run_dir):
    set_seed(cfg.seed)

    method = cfg.method.lower()
    target_mode = getattr(cfg, "target_mode", CT.DEFAULT_TARGET_MODE)
    seg_domain = getattr(cfg, "seg_domain", CT.DEFAULT_SEG_DOMAIN)

    installations = parse_int_list(cfg.train_installations, CT.DEFAULT_TRAIN_INSTALLATIONS)
    train_speeds = parse_int_list(cfg.train_speeds, CT.PROTOCOL_TRAIN_SPEEDS)
    val_speeds = parse_int_list(getattr(cfg, "val_speeds", None), CT.PROTOCOL_VAL_SPEEDS)
    target_speeds = parse_int_list(getattr(cfg, "target_speeds", None), CT.PROTOCOL_TEST_SPEEDS)
    target_installations = parse_int_list(
        getattr(cfg, "target_installations", None), installations)

    if set(train_speeds) & (set(val_speeds) | set(target_speeds)):
        raise ValueError("Source, validation and target speed bands must be disjoint")

    tr_loader, tr_data = build_source_loader(
        installations=installations,
        speeds=train_speeds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_segments_per_file=cfg.max_segments_per_file,
        seg_domain=seg_domain,
        seed=cfg.seed,
    )
    va_loader, _ = build_plain_loader(
        installations=installations,
        speeds=val_speeds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_segments_per_file=cfg.max_segments_per_file,
        seg_domain=seg_domain,
        seed=cfg.seed,
    )

    n_domains = tr_data.n_domains
    if not len(tr_loader) or not len(va_loader):
        raise ValueError("Training and validation loaders must be nonempty")
    needs_domains = method in ("dann", "coral", "msdal")
    if target_mode == "none" and needs_domains and n_domains < 2:
        raise ValueError(
            f"method={method} with target_mode='none' aligns the source sub-domains "
            f"against each other, which needs >= 2 of them; got {n_domains}. "
            f"Use more train speeds/installations, or target_mode='unlabeled'."
        )

    target_stream = None
    if target_mode == "unlabeled" and method != "source_only":
        te_loader, _ = build_plain_loader(
            installations=target_installations,
            speeds=target_speeds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            max_segments_per_file=cfg.max_segments_per_file,
            seg_domain=seg_domain,
            seed=cfg.seed,
            snr_db=getattr(cfg, "target_snr_db", None),
            shuffle=True,
            drop_last=True,
        )
        if not len(te_loader):
            raise ValueError("Target loader is empty; reduce batch_size or increase the data cap")
        target_stream = infinite(te_loader)

    model = build_model(
        method=method,
        num_classes=CT.NUM_CLASSES,
        embed_dim=cfg.embed_dim,
        channels=cfg.backbone_channels,
        disc_hidden=cfg.disc_hidden,
        n_domains=n_domains,
        target_mode=target_mode,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(run_dir, exist_ok=False)
    ckpt_path = os.path.join(run_dir, "state_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    steps_per_epoch = max(len(tr_loader), 1)
    total_steps = max(cfg.epochs * steps_per_epoch, 1)
    print(f"[train] method={method} target_mode={target_mode} "
          f"domains={tr_data.domain_ids_sorted} "
          f"source_n={len(tr_data)} steps/epoch={steps_per_epoch}")

    try:
        best_val, global_step = -1.0, 0
        for epoch in range(cfg.epochs):
            model.train()
            running = {"loss": 0.0, "cls": 0.0, "adapt": 0.0, "extra": 0.0}
            n_batches = 0

            for x_s, y_s, d_s in tr_loader:
                x_s = x_s.to(device)
                y_s = y_s.to(device)
                d_s = d_s.to(device)
                # Warm-up coefficient: near 0 while the features are still random,
                # ~1 by the end of training.
                lambd = ramp_lambda(global_step / total_steps, cfg.ramp_gamma) if cfg.ramp else 1.0

                feat_t = None
                if target_stream is not None:
                    x_t, _, _ = next(target_stream)
                    feat_t = model.features(x_t.to(device))

                feat_s = model.features(x_s)
                extra = torch.zeros((), device=device)

                if method == "msdal":
                    agg, cls_l, adapt_l, disc = _msdal_step(
                        model, feat_s, y_s, d_s, feat_t, lambd, cfg)
                    loss = agg + cfg.lambda_msdal_disc * lambd * disc
                    extra = disc
                else:
                    cls_l = F.cross_entropy(model.classify(feat_s), y_s)
                    if method == "source_only":
                        adapt_l = torch.zeros((), device=device)
                        loss = cls_l
                    elif method == "coral":
                        adapt_l = (coral_loss(feat_s, feat_t) if feat_t is not None
                                   else pairwise_coral_loss(feat_s, d_s))
                        loss = cls_l + cfg.lambda_coral * lambd * adapt_l
                    elif method == "dann":
                        adapt_l = _dann_step(model, feat_s, d_s, feat_t, lambd, model.binary)
                        loss = cls_l + cfg.lambda_dann * adapt_l
                    else:
                        raise ValueError(f"unhandled method={method!r}")

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                running["loss"] += float(loss.item())
                running["cls"] += float(cls_l.item())
                running["adapt"] += float(adapt_l.item())
                running["extra"] += float(extra.item())
                n_batches += 1

                writer.add_scalar("train/loss_total", float(loss.item()), global_step)
                writer.add_scalar("train/loss_cls", float(cls_l.item()), global_step)
                writer.add_scalar("train/loss_adapt", float(adapt_l.item()), global_step)
                writer.add_scalar("train/ramp_lambda", lambd, global_step)
                global_step += 1

            val_acc = validate(model, va_loader, device)
            avg = {k: v / max(n_batches, 1) for k, v in running.items()}
            writer.add_scalar("epoch/loss_total", avg["loss"], epoch + 1)
            writer.add_scalar("epoch/loss_cls", avg["cls"], epoch + 1)
            writer.add_scalar("epoch/loss_adapt", avg["adapt"], epoch + 1)
            writer.add_scalar("epoch/val_acc", val_acc, epoch + 1)
            extra_str = f" disc={avg['extra']:.4f}" if method == "msdal" else ""
            # `adapt` is the RAW alignment term, before its lambda; CORAL's is
            # orders of magnitude smaller than the adversarial ones, hence :.4g.
            print(f"[epoch {epoch + 1:03d}] loss={avg['loss']:.4f} cls={avg['cls']:.4f} "
                  f"adapt={avg['adapt']:.4g}{extra_str} val_acc={val_acc:.4f}")

            if val_acc > best_val:
                best_val = val_acc
                save_checkpoint(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": vars(cfg),
                        "method": method,
                        "target_mode": target_mode,
                        "n_domains": n_domains,
                        "domain_ids": tr_data.domain_ids_sorted,
                        "train_installations": installations,
                        "train_speeds": train_speeds,
                        "val_speeds": val_speeds,
                        "target_speeds": target_speeds if target_stream is not None else [],
                        "seg_domain": seg_domain,
                        "val_acc": best_val,
                        "classes": CT.CLASSES,
                        "epoch": epoch + 1,
                    },
                    ckpt_path,
                )

        print(f"[train] best val_acc={best_val:.4f} -> {ckpt_path}")
        return ckpt_path
    finally:
        writer.close()


def default_config(method="dann", *, installation=1, target_mode="unlabeled",
                   data_root=None, seed=42, device="cuda"):
    """Return editable settings; defaults reproduce the archived baseline setup."""
    if method not in CT.METHODS or target_mode not in CT.TARGET_MODES:
        raise ValueError("Unknown baseline method or target mode")
    if installation not in (1, 2):
        raise ValueError("installation must be 1 or 2")
    return types.SimpleNamespace(
        method=method, target_mode=target_mode, data_root=str(data_root or CT.DATA_ROOT),
        train_installations=[installation], train_speeds=[20, 25, 30, 35],
        val_speeds=[40], target_installations=[installation], target_speeds=[45, 50, 55],
        target_snr_db=None, seg_domain="time", max_segments_per_file=160,
        epochs=10, batch_size=64, num_workers=0, embed_dim=256, backbone_channels=64,
        disc_hidden=256, lr=0.001, weight_decay=1e-4, grad_clip=5.0,
        ramp=1, ramp_gamma=10.0, lambda_dann=1.0, lambda_coral=1e5,
        lambda_msdal_adv=1.0, lambda_msdal_disc=1.0, msdal_agg="mean", msdal_gamma=10.0,
        device=device, seed=seed,
        reference_protocol="disjoint20_random3_v1",
    )
