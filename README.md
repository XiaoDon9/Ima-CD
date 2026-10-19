# Cross-domain diagnostics via imagination

WT planetary-gearbox implementation of healthy-reference-conditioned fault
feature imagination. The default is the full Ima-CD model with MI and
map-diversity regularization. Both terms can be ablated individually or
together. Ima-CD--MLP is a nonlinear mapping variant; explicit controls
also test healthy alignment, feature conditioning and the healthy anchor.
Diagnostic baselines are provided separately.

No recordings, trained weights, experiment outputs, TensorBoard events,
historical result tables, sweep launchers or machine-specific configuration
are included. Training and evaluation are exposed as Python APIs.

## Structure

```text
imacd\
  config.py        WT protocol and experiment configuration
  data.py          MAT reading, segmentation and data loaders
  pairing.py       Same-domain reference/query sampling and normalization
  backbone.py      Shared encoder implementation
  model.py         Linear/MLP imagination, fixed-feature and all-class controls
  losses.py        Classification, healthy alignment, MINE critic and map diversity
  training.py      Single-experiment training and checkpoint evaluation
  utils.py         Seeds, checkpoints and metrics
  noise.py         Optional test-time signal noise
baselines\
  paired.py        Query CNN, concatenation and Siamese controls
  models.py        Source-only CNN, DANN, Deep CORAL and MSDAL
  losses.py        Domain adaptation objectives
  data.py          Single-signal loaders and domain-balanced batches
  config.py        Baseline protocol constants
  training.py      Baseline configuration and training API
  evaluation.py    Checkpoint-based baseline evaluation
tests\             Synthetic model, data and train/evaluate tests
requirements.txt
```

## Environment and data

Tested with Python 3.9.23, PyTorch 2.0.1+cu118, CUDA 11.8,
NumPy 1.25.1, SciPy 1.10.1 and TensorBoard 2.12.1.
Install a PyTorch build suitable for your hardware and the dependencies in
`requirements.txt`. Use `device="cpu"` explicitly when CUDA is unavailable.

Obtain the WT recordings independently:
https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset

Dataset reference: D. Liu, L. Cui and W. Cheng, *Measurement Science and
Technology* 35, 012002, https://doi.org/10.1088/1361-6501/acf390.
Dataset ownership and redistribution terms remain with its providers.

Pass `data_root` to the configuration, or set `WT_DATA_ROOT` before importing.
Expected directory layout:

```text
WT\
  healthy\1\*_20.MAT, ..., *_55.MAT
  healthy\2\*_20.MAT, ..., *_55.MAT
  broken\1\...          (each class has installations 1 and 2)
  missing_tooth\1\...
  root_crack\1\...
  wear\1\...
```

The MAT variable `Data` contains vibration in zero-based column 1.
The default uses non-overlapping 4096-point time windows, capped at
160 windows per recording. Class 0 is healthy.

## Model and experimental protocol

Each query is paired with three healthy reference windows from the same
installation and speed. Each input is centered by its own mean and divided
by the standard deviation of the concatenated **raw references**, plus
1e-6. Query amplitude does not determine this scale.

The reference encoder outputs are averaged and L2-normalized. Class-specific
linear maps imagine fault features from this healthy representation.
Cosine similarity between the query embedding and candidate class features
provides the classification scores. Each map is independently orthogonally
initialized and subsequently trained without an orthogonality constraint.

**Ima-CD--MLP** uses `Experiment(mapping="mlp")`: each fault matrix is
replaced by an independent `Linear(256,256) -> ReLU -> Linear(256,256)`.
Both weight matrices are orthogonally initialized and both biases are zero.
The healthy anchor, encoders, healthy scaling, classification and all three
auxiliary loss weights remain unchanged. Diversity uses the complete flattened
weights **and biases of both layers** per fault MLP. This variant tests
nonlinear mapping capacity; it is not a removal of a loss and is reported
alongside linear Ima-CD in the manuscript's comparison tables.

| Setting | Full Ima-CD default |
|---|---|
| Sources / labeled validation / test | 20,25,30,35 Hz / 40 Hz / 45,50,55 Hz |
| Installations | 1 and 2, trained separately |
| Repeated seeds | 42--51 |
| Encoder | Three Conv1D stages, widths 64/128/256, embedding 256 |
| Normalization / encoder sharing | BatchNorm / independent reference and query encoders |
| Budget | 10 epochs, batch size 64, workers 0 |
| Optimizer | Adam, LR 0.0003, cosine decay to 0.00001 |
| Weight decay / gradient clipping | 0.0001 / 5 |
| Classification temperature | 0.2 |
| Healthy alignment weight | 0.5 |
| MI weight / critic hidden width | 0.02 / 256 |
| Diversity weight / margin | 5 / sqrt(2) |
| Selection | Highest validation accuracy; latest epoch wins ties |

The default epoch contains 3200 training episodes. With 160 windows per
recording, validation enumerates 768 queries and testing enumerates 2304.
The full training objective is
`CE(scores / 0.2) + 0.5 * healthy_alignment + 0.02 * negative_DV + 5 * diversity`.
Healthy alignment uses healthy queries only. The MINE-style critic receives
the query feature and its ground-truth candidate anchor; a random anchor
permutation supplies the marginal pairs. Its three linear layers have widths
512/256/256/1, with two ReLU activations. The finite-batch DV statistic uses
clamped marginal scores (-20 to 20) and a 1e-8 logarithm stabilizer.
It is a training regularizer, not a measured mutual-information result.

Diversity averages `relu(sqrt(2) - distance)` over unordered pairs of
L2-normalized flattened fault matrices, with 1e-12 inside the distance square
root. The critic is optimized jointly with the encoders and maps; gradient
clipping covers all these parameters. It is not needed for inference.

This is amortized reference-conditioned learning, not an inner-loop MAML
implementation. Target fault labels are not used for optimization or
checkpoint selection. Labeled intermediate-speed validation is used.
The corrected protocol reserves 20% of healthy windows in each domain
(32 of 160) as a fixed reference-only pool. Its partition uses seed
20261018 plus the domain ID, independently of optimization seeds 42--51.
The remaining healthy windows form the healthy query set; fault queries
are not removed. Every query draws three references uniformly with
replacement from its domain's fixed pool. References may repeat within
the triplet, but a healthy query cannot be its own reference.
Training resamples triplets; validation/test use per-query seeded draws,
reproducible across epochs and access order. All baselines exclude the
same reference-only healthy windows from their query sets.
The recordings are still from the same acquisition, not independent
commissioning sessions. Old overlapping-pool checkpoints are rejected
by the corrected evaluation API and require their archived loaders.

## Reproducing the model and ablations

Use the same API for every variant; there are no separate execution scripts.
From the repository root, the following Python example trains one run:

```python
from pathlib import Path
import torch
from imacd import Experiment, train, evaluate

torch.set_num_threads(2)
torch.set_num_interop_threads(2)  # Set once, before parallel work starts.
cfg = Experiment(data_root=r"D:\datasets\WT", variant="full",
                 installation=1, seed=42, device="cuda")
result = train(cfg, output_dir=r"outputs\imacd")
checkpoint = Path(r"outputs\imacd").joinpath(cfg.name, "best.pt")
metrics = evaluate(checkpoint,
                   data_root=r"D:\datasets\WT", device="cuda")
```

`train` creates one descriptive run directory with best/last checkpoints,
metrics and TensorBoard logs. Existing runs are never overwritten.
`evaluate` reconstructs the architecture and preprocessing from the saved
configuration. Repeat independently over both installations and seeds42--51;
report mean and **sample standard deviation** of accuracy and macro-F1.
Changing the budget or architecture is a new experiment, not reproduction
of the default configuration.

| Variant | MI weight | Diversity weight |
|---|---|---|
| full (default) | 0.02 | 5 |
| no_mi | 0 | 5 |
| no_diversity | 0.02 | 0 |
| no_both | 0 | 0 |

Select an ablation with, for example, `Experiment(variant="no_mi")`.
These four settings default to fault-only linear maps, the unchanged healthy
anchor, healthy alignment (0.5), and identical data/optimization settings.
The earlier S0--S4 architecture identifiers are no longer supported.

### What each design control changes

`mapping` selects candidate generation; `variant` selects the learning-objective
control. The two fields are separate so mapping and alignment are not
implicitly changed together. `variant="no_alignment"` disables only healthy
alignment, retaining MI0.02 and diversity5.

| Configuration | Design being tested | Reference for a direct comparison |
|---|---|---|
| `mapping="linear", variant="full"` | Full linear Ima-CD | Main model |
| `mapping="mlp", variant="full"` | Nonlinear healthy-to-fault relation | Full linear Ima-CD |
| `variant="no_mi"` | Query/correct-anchor dependency objective | Full linear Ima-CD |
| `variant="no_diversity"` | Separation of class-map parameter directions | Full linear Ima-CD |
| `variant="no_both"` | Joint auxiliary-regularizer removal | Full linear Ima-CD |
| `variant="no_alignment"` | Healthy query/reference agreement | Full linear Ima-CD |
| `mapping="fixed", variant="no_alignment"` | Condition-dependent candidate input | Linear model without alignment |
| `mapping="all_class", variant="no_alignment"` | Unmapped healthy anchor | Linear model without alignment |
| `mapping="fixed", variant="full"` | Condition-dependent candidate input, with alignment retained | Full linear Ima-CD |
| `mapping="all_class", variant="full"` | Unmapped healthy anchor, with alignment retained | Full linear Ima-CD |

The fixed-feature control learns a normalized constant vector `v` and generates
`[v, v W1, ...]`, independent of reference features. "Fixed" means constant
across conditions at inference, not frozen during training. Healthy-reference
input scaling is still active. With alignment disabled, the unused reference
encoder is omitted, matching the recorded fixed-feature experiment.

All-class mapping instead generates `[r W0, r W1, ...]` from the measured
healthy reference feature `r`. It adds a healthy-state map rather than
removing healthy conditioning; diversity covers **all** class matrices.
The alignment-free fixed-feature and all-class rows disable alignment. Their
direct comparator is the no-alignment linear model, not the full model:
otherwise two design changes would be conflated.

Alignment-preserving controls are available as
`Experiment(mapping="fixed", variant="full")` and
`Experiment(mapping="all_class", variant="full")`. Both are benchmarked in
the manuscript, along with their alignment-free counterparts. Alignment always
compares a healthy query with the **raw, unmapped measured reference feature**,
never the fixed seed or mapped healthy anchor. The aligned fixed-feature
control therefore retains a reference encoder solely for alignment while its
candidate generator remains independent of healthy features. In evaluation
mode, changing reference features cannot change its candidate bank, although
healthy-based input scaling still affects the preprocessed query.
These definitions avoid changing the meaning of alignment between controls.

Single-term ablations retain the critic, including its permutation and
optimizer parameters, matching the original full-model ablation protocol.
In `no_mi`, its loss weight is zero: data gradients are zero, although Adam
weight decay can still update critic parameters. `no_both` actually omits
the critic and diversity computation, matching the recorded joint-removal
experiment. Its changed random-number consumption means the joint-removal
comparison describes complete training configurations, not a numerically
identical stochastic trajectory with two coefficients changed.

TensorBoard run names include the mapping, variant and alignment/MI/diversity weights. Batch
and epoch records include classification, alignment, MI (the **negative**
DV statistic), diversity and total loss. Disabled diversity is logged as
zero; in `no_mi`, the critic's unweighted statistic is still logged.
Best/last checkpoints include critic and optimizer states when applicable;
evaluation restores only the diagnostic model, without a critic.

Normalization/sharing controls use `Experiment(encoder_norm="group")`,
`Experiment(encoder_sharing="shared")` or
`Experiment(input_normalization="window")`, changing one factor at a time.
Query, concatenation and absolute-difference Siamese controls use
`Experiment(model_type="query")`, `"concat"` or `"siamese"` with the same
paired data protocol and optimizer; these classifiers use ordinary
cross-entropy without temperature scaling or alignment.
To reproduce the manuscript's WT query and concatenation comparison rows,
set `input_normalization="window", lr=0.001` explicitly; their other settings
retain the ten-epoch cosine schedule and latest-best checkpoint rule.
The default paired-control API otherwise uses healthy scaling and LR0.0003.

## Reproducing transfer baselines

```python
from baselines import default_config, train, evaluate

cfg = default_config("dann", data_root=r"D:\datasets\WT",
                     installation=1, target_mode="none", seed=42)
checkpoint = train(cfg, output_dir=r"outputs\baselines")
metrics = evaluate(checkpoint, data_root=r"D:\datasets\WT", device="cuda")
```

Methods are `source_only`, `dann`, `coral` and `msdal`.
Baseline defaults preserve their earlier protocol: window-wise z-scoring,
BatchNorm, 10 epochs, batch64, embedding256, Adam LR0.001,
weight decay0.0001 and clipping5, **without cosine scheduling**.
Checkpoint selection keeps the **first** validation-best epoch on ties.
DANN weight1 uses the progress ramp with gamma10; CORAL weight is100000;
MSDAL adversarial/discrepancy weights are1 with mean source aggregation.
The per-source classifiers and binary discriminators implement a local
multi-source adversarial baseline, not an external MSDAL release.

`target_mode="none"` uses source domains only during optimization.
`target_mode="unlabeled"` exposes unlabeled target-speed windows during
adaptation and updates BN with them. Source-only ignores the target stream
in either mode. These information budgets differ from inference-time
healthy-reference conditioning, and the recorded baseline optimization and
normalization differ from the full model's defaults. Do not describe
them as strictly matched single-factor ablations.

## Tests and publication

```powershell
python -m unittest discover -s tests -v
```

Tests generate temporary synthetic MAT files; no real recordings are needed.
They cover the regularization settings, linear/MLP/fixed/all-class mappings
with alignment on/off, loss formulas and gradients,
normalization/sharing, paired classifiers, all four transfer
methods in both information modes, and training/checkpoint/evaluation round
trips. Importing the packages does not create output directories.

This source-only release supersedes the earlier flat bundle. Old checkpoints
are not directly interchangeable with the reorganized APIs; local original
experiment folders remain separate. This release restores the full model
as the default without adding training outputs or sweep launchers. Before
publishing, the rights holders should choose
a software license; no license or repository URL is invented here.
