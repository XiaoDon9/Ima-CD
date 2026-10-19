# Cross-domain diagnostics via imagination

PyTorch implementation of **Ima-CD** for cross-domain fault diagnosis on the
WT planetary-gearbox dataset, including ablations and diagnostic baselines.

## Setup

Tested with Python 3.9 and PyTorch 2.0.1. Install a PyTorch build suitable
for your hardware, then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

Download the [WT dataset](https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset)
separately and organize it as:

```text
WT\
  healthy\1\*.MAT
  healthy\2\*.MAT
  broken\1\*.MAT
  broken\2\*.MAT
  missing_tooth\...
  root_crack\...
  wear\...
```

Each class has installations `1` and `2`, with recordings at 20--55 Hz
in 5 Hz steps. Files use the `*_20.MAT`, ..., `*_55.MAT` naming pattern;
vibration is read from the second column of `Data`.
Raw data, trained weights and experiment outputs are not included.

## Train and evaluate

Run from the repository root, replacing `data_root` with your dataset path:

```python
from pathlib import Path
from imacd import Experiment, train, evaluate

cfg = Experiment(data_root=r"D:\datasets\WT",
                 installation=1, seed=42, device="cuda")
train(cfg, output_dir=r"outputs\imacd")
checkpoint = Path(r"outputs\imacd") / cfg.name / "best.pt"
metrics = evaluate(checkpoint, data_root=cfg.data_root, device=cfg.device)
print(metrics)
```

Use `device="cpu"` when CUDA is unavailable.
Defaults: 10 epochs, batch size 64, Adam learning rate 0.0003.
Training uses 20/25/30/35 Hz, labeled validation uses 40 Hz, and testing
uses 45/50/55 Hz. Each query uses three same-domain healthy references
from a fixed reference-only pool disjoint from healthy queries.
For repeated experiments, use installations `1` and `2` and seeds `42`--`51`.

Set `variant` to `full` (default), `no_mi`, `no_diversity`, `no_both` or
`no_alignment`. Use `mapping="mlp"` for Ima-CD--MLP.
Additional settings are in [imacd/config.py](imacd/config.py).

## Baselines

```python
from baselines import default_config, train, evaluate

cfg = default_config("dann", data_root=r"D:\datasets\WT",
                     installation=1, target_mode="none", seed=42)
checkpoint = train(cfg, output_dir=r"outputs\baselines")
metrics = evaluate(checkpoint, data_root=cfg.data_root, device="cuda")
print(metrics)
```

Available methods: `source_only`, `dann`, `coral` and `msdal`.
`target_mode="none"` excludes target data from training;
`"unlabeled"` enables unlabeled target adaptation for adaptation methods.
Baseline normalization and optimization settings differ from Ima-CD.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests check code functionality using synthetic data, without WT recordings.
Reproducing the paper's experiments requires the real WT dataset.

## License

Code is licensed under [MIT](LICENSE): use, modification and redistribution
are permitted with the copyright and license notices retained, without warranty.
The WT dataset is subject to its providers' terms.
