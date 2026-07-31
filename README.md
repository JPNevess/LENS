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

## Reproducing the figures

Every figure and the results table are built from the CSVs committed in
`results/`, so no experiment has to be re-run:

```bash
cd figures
python make_all.py
```

Output goes to `figures/output/`. Each script is standalone and states at the top
which figure it draws and which file it reads.

## Running a configuration

Each file in `benchmarks/` is a standalone evaluation of one configuration. It
states the mechanism, spells out the protocol constants it runs under,
implements the selection rule, and has its own command line. Nothing has to be
read alongside it to know what was run.

The protocol constants are deliberately identical across the files: same base
learners, same ensemble size, same streams, same label rates, same seeds, same
prequential test-then-train evaluation. That is what lets the table be read as a
comparison of mechanisms — a difference between two rows is a difference in
selection or self-training, not in tuning or in implementation effort.

```bash
python benchmarks/run_lens.py                        # the full method
python benchmarks/ablation_mmr_selection.py          # one cell of the grid
python benchmarks/run_lens.py --datasets Electricity --label-pcts 5
```

Rows are appended to `results/benchmarks/runs.csv`, one per run, and the script
prints the cell means at the end.

### What these files are, and what they are not

**The `ablation_*` files are cells of this repository's own factorial ablation,
not implementations of published methods.** Each is the LENS ensemble with the
selection and self-training axes set to one combination, and each is named after
the mechanism it switches on — `Margin`, `RefereeAcc`, `MMR`, `SelfTrain-M` and
so on. Several of them isolate an idea that a published method is built around,
and the file says so in prose, but none of them is a port of that method and no
row of the results table should be read as a measurement of it.

`sleade_implementation.py` is the one exception. It runs the SLEADE
implementation shipped with CapyMOA, through CapyMOA's own evaluator, so it is
the single external point of comparison in the table.

Every reported number is the mean over five seeds, which is the default. A seed
initialises the learners and, offset by `LABEL_SEED_OFFSET`, also draws which
instances arrive labelled; on the noisier streams a single seed sits more than a
point away from that mean, so one seed alone is not comparable with the reported
values.

## Re-running the studies

`experiments/` holds the runs that produce `results/`. Each one resumes from what
is already in its output file, so it can be stopped and restarted.

| script | produces | used by |
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
large and are not committed; `export_figure_data.py` reduces them to the compact
tables in `results/figure_data/`, which are:

```bash
python experiments/export_figure_data.py --source results/probes
```

## Layout

```
lens/           the method: ensemble, referee, evaluation loop, stream generators
benchmarks/     one standalone script per configuration in the comparison
experiments/    the studies that produce results/
figures/        one script per figure, reading only results/
results/        committed CSVs the figures are built from
data/           the real streams, and the cache for generated ones
third_party/    vendored MLHAT reference implementation
```

## Streams

The synthetic streams are not shipped as files. They are defined by
`lens/streams.py` and generated the first time a run asks for one, then cached
under `data/`; the definition is what is committed, not the bytes. The real
streams cannot be generated and have to be present in `data/` — see
`data/README.md`.

```bash
python experiments/make_datasets.py                 # build them all ahead of time
python experiments/make_datasets.py --datasets LED_a RBF_m --force
```

Two seeds decide what a run sees, and they are kept apart on purpose:
`STREAM_SEED` in `lens/streams.py` fixes the concept geometry, the drift
transitions and the instance draw, while `label_seed` decides which instances
arrive with their label revealed. Both synthetic and real streams have a
`label_seed`; only synthetic ones have a `STREAM_SEED`.

### Provenance of the committed results

The CSVs under `results/` were produced from an earlier draw of the synthetic
streams, before the drift geometry in `lens/streams.py` was changed: transitions
are now wider, abrupt change points are irregularly spaced rather than evenly
spaced, the seeds differ, and the labelled subset is now drawn rather than taken
at a fixed period. **Regenerating the streams and re-running will therefore not
reproduce the numbers in `results/` exactly.** The figures are built from the
committed CSVs, so they are unaffected; anyone re-running the studies from
scratch should expect different values and should re-run every configuration
rather than comparing new runs against the committed ones.

## Third-party code

`third_party/mlhat` is the reference implementation of Hoeffding adaptive trees
for multi-label classification (Esteban et al., *Knowledge-Based Systems* 304,
2024), used unmodified as the referee. Its own licence applies.
