# LENS

LENS is a semi-supervised ensemble for evolving data streams. A lightweight
multi-target meta-learner, the referee, predicts per instance whether each
ensemble member is currently correct. Because it models the members jointly
rather than one at a time, the same estimates serve three mechanisms that are
usually built separately:

* **selection** — members are chosen by maximal marginal relevance, trading
  estimated competence against meta-learned diversity;
* **self-training** — pseudo-labels for unlabelled instances are admitted and
  weighted by meta-learned reliability;
* **drift adaptation** — ADWIN monitors the same signals, and after a change the
  trade-off shifts towards diversity while weak members are replaced from a
  background pool.

This repository contains the code and nothing else. No stream and no measurement
is committed: everything is produced by running the scripts below.

## Install

Requires Python 3.10+ and a JDK 11 or newer, which CapyMOA needs to start a JVM.

```bash
pip install -r requirements.txt
```

`JAVA_HOME` is located automatically from the usual install locations. If no JDK
is found, set it explicitly:

```bash
# macOS
export JAVA_HOME=$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home
# Linux
export JAVA_HOME=/usr/lib/jvm/default-java
```

## Streams

The synthetic streams are generated, one script each, and cached under `data/`.
The real ones have to be downloaded into the same directory. See
[`data/README.md`](data/README.md) for the full list and where the real ones come
from.

```bash
python data/make_led_a.py          # one stream
python experiments/make_datasets.py   # all of the synthetic ones
```

Generating ahead of time is optional: a run builds the stream it needs on first
use.

Two seeds decide what a run sees, and they are kept apart on purpose.
`STREAM_SEED` in `lens/streams.py` fixes the concept geometry, the drift
transitions and the instance draw; `label_seed` decides which instances arrive
with their label revealed. Both synthetic and real streams have a `label_seed`;
only synthetic ones have a `STREAM_SEED`. Changing either means results produced
before and after are not comparable.

## Running a method

Each file in `benchmarks/` is a standalone evaluation of one row of the
comparison, named after the method that row reports. It states the mechanism,
spells out the protocol constants it runs under, implements the selection rule,
and has its own command line. Nothing has to be read alongside it to know what
was run.

```bash
python benchmarks/lens.py                            # the full method
python benchmarks/dyned.py                           # one baseline
python benchmarks/lens.py --datasets Electricity --label-pcts 5
```

| file | row | file | row |
|---|---|---|---|
| `arf.py` | ARF | `lst.py` | LST* |
| `dyabst.py` | DyAbst | `sco_for.py` | SCo-For |
| `meta_br.py` | Meta-BR | `sleade.py` | SLEADE |
| `dems.py` | DEMS | `lens_m.py` | LENS-M |
| `dyned.py` | DynED | `lens.py` | LENS |
| `self_train.py` | Self-train | | |

Rows are appended to `results/benchmarks/runs.csv`, one per run, and the script
prints the cell means at the end.

The protocol constants are deliberately identical across the files: same base
learners, same ensemble size, same streams, same label rates, same seeds, same
prequential test-then-train evaluation. That is what lets the table be read as a
comparison of mechanisms — a difference between two rows is a difference in
selection or self-training, not in tuning or in implementation effort.

Every reported number is the mean over five seeds, which is the default. A seed
initialises the learners and, offset by `LABEL_SEED_OFFSET`, also draws which
instances arrive labelled; on the noisier streams a single seed sits more than a
point away from that mean, so one seed alone is not comparable with a reported
value.

`sleade.py` runs the SLEADE implementation shipped with CapyMOA, through
CapyMOA's own evaluator. The other rows are implemented in this repository's
ensemble so that the protocol is shared, and each file records under *How this
row is built* the configuration it runs and what it takes from the method it is
named after.

## Studies

`experiments/` holds the runs behind the figures. Each resumes from what is
already in its output file, so it can be stopped and restarted.

| script | produces | feeds |
|---|---|---|
| `run_ablation.py` | the factorial grid | Table 1, Figures 4, 5, 6, 8 |
| `run_seeds.py` | the same grid over several seeds | the spread in Table 1 |
| `run_diversity_study.py` | diversity estimates vs ground truth | Figure 1 (top) |
| `run_lambda_study.py` | the relevance/diversity sweep | Figures 1, 10a, 10c |
| `run_referee_ablation.py` | high-order vs binary relevance referee | Figures 7, 11 |
| `run_admission_study.py` | pseudo-label admission signals | Figure 9 |
| `run_drift_detection.py` | drift detection signals | Figure 10d |
| `compute_robustness_metrics.py` | pairwise diagnostics | Figures 5b, 5c |
| `compute_signal_importance.py` | variance decomposition | Figure 5a |

The studies write per-instance probe archives under `results/probes/`. Those are
large; `export_figure_data.py` reduces them to the compact tables the figures
read:

```bash
python experiments/export_figure_data.py --source results/probes
```

## Figures

`figures/` holds one script per figure. They read only from `results/`, so the
studies above have to have been run first.

```bash
python figures/make_all.py
```

Output goes to `figures/output/`. Each script is standalone and states at the top
which figure it draws and which file it reads.

## Layout

```
lens/           the method: ensemble, referee, evaluation loop, stream definitions
benchmarks/     one standalone script per row of the comparison
experiments/    the studies behind the figures
figures/        one script per figure
data/           one generator per synthetic stream; downloaded streams live here
third_party/    vendored MLHAT reference implementation
```

`results/`, `figures/output/` and the stream files are produced by running the
code and are not tracked.

## Third-party code

`third_party/mlhat` is the reference implementation of Hoeffding adaptive trees
for multi-label classification (Esteban et al., *Knowledge-Based Systems* 304,
2024), used unmodified as the referee. Its own licence applies.
