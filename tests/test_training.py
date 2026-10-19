import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from imacd import Experiment, evaluate, train
from imacd import config
from imacd.utils import load_checkpoint
from baselines import default_config, train as train_baseline, evaluate as evaluate_baseline
from baselines.data import clear_cache, load_flat
from baselines import config as baseline_config

torch.set_num_threads(2)


def make_recordings(root, offset=0):
    rng = np.random.default_rng(7)
    for cls in config.CLASSES:
        directory = root / cls / "1"
        directory.mkdir(parents=True)
        for speed in (20, 25, 30, 35, 40, 45, 50, 55):
            signal = rng.normal(size=2 * 4096).astype(np.float32) + offset
            sio.savemat(directory / f"test_1_{speed}.MAT",
                        {"Data": np.column_stack([np.zeros_like(signal), signal])})


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        make_recordings(cls.data)

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        cls.temp.cleanup()

    def test_regularization_and_paired_train_evaluate_roundtrips(self):
        cases = ([("imacd", "linear", v) for v in
                  ("full", "no_mi", "no_diversity", "no_both", "no_alignment")]
                 + [("imacd", m, v) for m in ("mlp", "fixed", "all_class")
                    for v in ("full", "no_alignment")]
                 + [(name, "linear", "full") for name in ("query", "concat", "siamese")])
        for method, mapping, variant in cases:
            with self.subTest(method=method, mapping=mapping, variant=variant):
                cfg = Experiment(data_root=str(self.data), variant=variant, model_type=method,
                                 mapping=mapping,
                                 epochs=1, batch_size=64, embed_dim=8, backbone_channels=4,
                                 max_segments_per_file=2, mine_hidden=8, device="cpu")
                output = self.root / "paired"
                result = train(cfg, output)
                checkpoint = output / cfg.name / "best.pt"
                restored = evaluate(checkpoint)
                self.assertEqual(restored, result["test"])
                self.assertEqual(restored["n"], 27)
                state = load_checkpoint(checkpoint)
                self.assertEqual("mine_state_dict" in state, method == "imacd" and variant != "no_both")
                last = load_checkpoint(checkpoint.parent / "last.pt")
                for key in ("optimizer_state_dict", "scheduler_state_dict"):
                    self.assertIn(key, last)
                self.assertEqual(state["epoch"], 1)
                events = EventAccumulator(str(checkpoint.parent / "tb"))
                events.Reload()
                for term in ("mi", "diversity", "alignment", "classification"):
                    self.assertIn("train/" + term, events.Tags()["scalars"])
                    self.assertIn(term, result["history"][0])
                if variant == "no_both" or method != "imacd":
                    self.assertEqual(result["history"][0]["mi"], 0)
                    self.assertEqual(result["history"][0]["diversity"], 0)
                if variant == "no_alignment":
                    self.assertEqual(result["history"][0]["alignment"], 0)
                self.assertEqual(state["cfg"]["mapping"], mapping)
                with self.assertRaises(FileExistsError):
                    train(cfg, output)
                saved = json.loads((checkpoint.parent / "metrics.json").read_text())
                self.assertEqual(saved["test"], restored)

    def test_all_transfer_methods_both_information_modes(self):
        with redirect_stdout(io.StringIO()):
            for mode in ("none", "unlabeled"):
                for method in ("source_only", "dann", "coral", "msdal"):
                    with self.subTest(method=method, mode=mode):
                        cfg = default_config(method, data_root=self.data, target_mode=mode, device="cpu")
                        cfg.epochs, cfg.batch_size, cfg.max_segments_per_file = 1, 8, 2
                        cfg.embed_dim, cfg.backbone_channels, cfg.disc_hidden = 8, 4, 8
                        checkpoint = train_baseline(cfg, self.root / "transfer")
                        restored = evaluate_baseline(checkpoint)
                        self.assertEqual(restored["n"], 27)
                        self.assertTrue(0 <= restored["accuracy"] <= 1)
                        self.assertTrue(0 <= restored["macro_f1"] <= 1)
                        state = load_checkpoint(checkpoint)
                        self.assertEqual(state["cfg"]["lr"], 0.001)
                        self.assertEqual(state["target_mode"], mode)
                        self.assertEqual(state["n_domains"], 4)
                        self.assertEqual(restored, evaluate_baseline(checkpoint, data_root=self.data))
                        with self.assertRaises(FileExistsError):
                            train_baseline(cfg, self.root / "transfer")

    def test_dataset_cache_is_keyed_by_root(self):
        second = self.root / "second"
        make_recordings(second, offset=100)
        with redirect_stdout(io.StringIO()):
            with baseline_config.data_settings(self.data):
                first = load_flat([1], [20], max_segments_per_file=2)
            with baseline_config.data_settings(second):
                other = load_flat([1], [20], max_segments_per_file=2)
        self.assertGreater(float(other.x.mean() - first.x.mean()), 90)

    def test_reject_missing_data_and_invalid_baseline(self):
        cfg = Experiment(data_root=str(self.root / "missing"), device="cpu")
        previous = config.DATA_ROOT
        with self.assertRaises(FileNotFoundError):
            train(cfg, self.root / "missing-output")
        self.assertEqual(previous, config.DATA_ROOT)
        with self.assertRaises(ValueError):
            default_config("unknown")
        self.assertFalse((self.root / "missing-output").exists())
