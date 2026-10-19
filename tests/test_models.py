import unittest

import torch
import torch.nn.functional as F

from imacd import Experiment, ImaCD
from imacd.losses import MINE, healthy_alignment, map_diversity, objective
from imacd.training import build_model as build_imacd
from baselines.losses import coral_loss, ramp_lambda
from baselines.models import build_model, grad_reverse
from baselines.paired import PairedClassifier

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_regularization_models_and_gradients(self):
        for variant in ("full", "no_mi", "no_diversity", "no_both"):
            with self.subTest(variant=variant):
                cfg = Experiment(variant=variant, embed_dim=8, backbone_channels=8)
                model = ImaCD(5, 8, 8)
                critic = MINE(8, 8) if variant != "no_both" else None
                refs, query, y = torch.randn(5, 3, 128), torch.randn(5, 1, 128), torch.arange(5)
                out = model(refs, query)
                self.assertEqual(model.label_weights.shape, (4, 8, 8))
                for matrix in model.label_weights:
                    torch.testing.assert_close(matrix.T @ matrix, torch.eye(8), atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(out["anchors"][:, 0], out["z_n"])
                loss, terms = objective(out, y, model.label_weights, critic,
                                        mi_weight=cfg.mi_weight, diversity_weight=cfg.diversity_weight)
                torch.testing.assert_close(
                    loss, terms["classification"] + cfg.alignment_weight * terms["alignment"]
                    + cfg.mi_weight * terms["mi"] + cfg.diversity_weight * terms["diversity"])
                loss.backward()
                self.assertIsNotNone(model.label_weights.grad)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
                if critic is not None:
                    for parameter in critic.parameters():
                        self.assertIsNotNone(parameter.grad)
                        if not cfg.mi_weight:
                            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_regularizer_formulas(self):
        critic = MINE(8, 8)
        query, anchor = torch.randn(5, 8), torch.randn(5, 8)
        torch.manual_seed(71)
        actual = critic(query, anchor)
        torch.manual_seed(71)
        joint = critic.t(torch.cat([query, anchor], -1)).mean()
        marginal = critic.t(torch.cat([query, anchor[torch.randperm(5)]], -1))
        expected = -(joint - (marginal.clamp(-20, 20).exp().mean() + 1e-8).log())
        torch.testing.assert_close(actual, expected)
        matrices = torch.ones(3, 8, 8, requires_grad=True)
        diversity = map_diversity(matrices)
        self.assertAlmostEqual(diversity.item(), 2 ** .5 - 1e-6, places=5)
        diversity.backward()
        self.assertTrue(torch.isfinite(matrices.grad).all())
        self.assertEqual(map_diversity(matrices[:1]).item(), 0)
        cfg = Experiment()
        self.assertEqual((cfg.variant, cfg.mi_weight, cfg.diversity_weight), ("full", .02, 5.0))

    def test_mapping_variants_and_alignment_targets(self):
        refs, query, labels = torch.randn(5, 3, 128), torch.randn(5, 1, 128), torch.arange(5)
        for mapping in ("linear", "mlp", "fixed", "all_class"):
            for variant in ("full", "no_alignment"):
                with self.subTest(mapping=mapping, variant=variant):
                    cfg = Experiment(mapping=mapping, variant=variant,
                                     embed_dim=8, backbone_channels=8, mine_hidden=8)
                    model = build_imacd(cfg)
                    critic = MINE(8, 8)
                    output = model(refs, query)
                    if mapping == "mlp":
                        expected = torch.stack([torch.cat([p.reshape(-1) for p in transform.parameters()])
                                                for transform in model.fault_maps])
                        torch.testing.assert_close(model.diversity_parameters(), expected)
                        self.assertEqual(expected.shape, (4, 2 * 8 * 8 + 2 * 8))
                        for transform in model.fault_maps:
                            for layer in transform:
                                if isinstance(layer, torch.nn.Linear):
                                    torch.testing.assert_close(layer.weight.T @ layer.weight, torch.eye(8),
                                                               atol=1e-5, rtol=1e-5)
                                    self.assertEqual(layer.bias.count_nonzero().item(), 0)
                        torch.testing.assert_close(output["anchors"][:, 0], output["z_n"])
                    elif mapping == "all_class":
                        self.assertEqual(model.label_weights.shape, (5, 8, 8))
                        torch.testing.assert_close(output["anchors"],
                                                   torch.einsum("bd,cde->bce", output["z_n"],
                                                                model.label_weights))
                    elif mapping == "fixed":
                        model.eval()
                        first = model(refs, query)
                        other = model(refs * 2 + torch.randn_like(refs), query)
                        torch.testing.assert_close(first["anchors"], other["anchors"])
                        self.assertEqual(hasattr(model, "normal_encoder"), variant == "full")
                        if variant == "full":
                            self.assertFalse(torch.allclose(first["z_n"], other["z_n"]))
                            self.assertFalse(torch.allclose(first["z_n"], first["anchors"][:, 0]))
                    loss, terms = objective(output, labels, model.diversity_parameters(), critic,
                                            alignment_weight=cfg.alignment_weight)
                    if cfg.alignment_weight:
                        torch.testing.assert_close(terms["alignment"],
                                                   healthy_alignment(output["z_q"], output["z_n"], labels))
                    else:
                        self.assertEqual(terms["alignment"].item(), 0)
                    loss.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters()))
        names = [Experiment(mapping=m, variant=v).name for m in ("linear", "mlp", "fixed", "all_class")
                 for v in ("full", "no_alignment")]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all("ALIGN" in name for name in names))

    def test_alignment_definition_and_no_healthy_batch(self):
        out = dict(z_n=torch.randn(4, 8), z_q=torch.randn(4, 8),
                   logits=torch.randn(4, 5, requires_grad=True))
        y = torch.tensor([0, 1, 0, 2])
        expected = (1 - F.cosine_similarity(out["z_q"][[0, 2]], out["z_n"][[0, 2]], dim=-1)).mean()
        torch.testing.assert_close(healthy_alignment(out["z_q"], out["z_n"], y), expected)
        loss, terms = objective(out, torch.ones(4, dtype=torch.long), torch.randn(4, 8, 8),
                                mi_weight=0, diversity_weight=0)
        torch.testing.assert_close(loss, terms["classification"])
        self.assertEqual(terms["alignment"].item(), 0)
        with self.assertRaises(ValueError):
            objective(out, y, torch.randn(4, 8, 8))

    def test_normalization_and_sharing_controls(self):
        for norm in ("batch", "group"):
            for sharing in ("independent", "shared"):
                model = ImaCD(5, 8, 8, encoder_norm=norm, encoder_sharing=sharing)
                self.assertEqual(model.normal_encoder is model.query_encoder, sharing == "shared")
                kind = torch.nn.BatchNorm1d if norm == "batch" else torch.nn.GroupNorm
                self.assertTrue(any(isinstance(layer, kind) for layer in model.modules()))
                self.assertEqual(model(torch.randn(3, 3, 128), torch.randn(3, 1, 128))["logits"].shape, (3, 5))

    def test_transfer_and_paired_baselines(self):
        for mode in ("none", "unlabeled"):
            for method in ("source_only", "dann", "coral", "msdal"):
                model = build_model(method, 5, 8, channels=4, disc_hidden=8,
                                    n_domains=2, target_mode=mode)
                out = model(torch.randn(4, 1, 128))
                loss = F.cross_entropy(out["logits"], torch.arange(4))
                if method == "dann":
                    loss = loss + model.domain_logits(out["feat"], 1.0).square().mean()
                if method == "msdal":
                    loss = loss + sum(model.domain_logits(out["feat"], k, 1.0).square().mean()
                                      for k in range(2))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
        for method in ("query", "concat", "siamese"):
            model = PairedClassifier(method, 5, 8, 4)
            loss = model(torch.randn(4, 3, 128), torch.randn(4, 1, 128))["logits"].sum()
            loss.backward()

    def test_adaptation_losses(self):
        x = torch.randn(4, 8, requires_grad=True)
        grad_reverse(x, 0.5).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, -0.5))
        self.assertEqual(coral_loss(x, x).item(), 0)
        self.assertAlmostEqual(ramp_lambda(0), 0)
        self.assertGreater(ramp_lambda(1), 0.99)

    def test_reject_invalid_configuration(self):
        for kwargs in (dict(variant="missing"), dict(encoder_norm="other"),
                       dict(installation=3), dict(temperature=0), dict(num_workers=1),
                       dict(variant="no_mi", model_type="query"), dict(variant="S0"),
                       dict(mine_hidden=0), dict(mapping="missing"),
                       dict(mapping="mlp", model_type="query")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Experiment(**kwargs)
