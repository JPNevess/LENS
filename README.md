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

## Running a method

Each entry point in `benchmarks/` evaluates one method under the shared protocol
defined in `benchmarks/_runner.py`: same base learners, same ensemble size, same
prequential test-then-train evaluation, same streams. Only the selection and
self-training mechanisms differ.

```bash
python benchmarks/run_lens.py                       # the full method
python benchmarks/our_implementation_of_dyned.py    # one baseline
python benchmarks/run_lens.py --datasets Electricity --label-pcts 5 --seeds 42
```

Rows are appended to `results/benchmarks/runs.csv`.

The baselines named `our_implementation_of_*` reproduce the mechanism of a
published method inside this harness; they are not ports of the original
implementations. `sleade_implementation.py` is the exception: it runs the
reference implementation shipped with CapyMOA.

## Re-running the studies

`experiments/` holds the runs that produce `results/`. They write per-instance
probe archives, which are large and are not committed;
`experiments/export_figure_data.py` reduces them to the compact tables the
figures read.

```bash
python experiments/make_datasets.py --datasets LED_a RBF_m
python experiments/export_figure_data.py --source results/probes
```

## Layout

```
lens/           the method: ensemble, referee, evaluation loop
benchmarks/     one entry point per method in the comparison
experiments/    the studies that produce results/, and the dataset generator
figures/        one script per figure, reading only results/
results/        committed CSVs the figures are built from
data/           the streams (see data/README.md)
third_party/    vendored MLHAT reference implementation
```

## Third-party code

`third_party/mlhat` is the reference implementation of Hoeffding adaptive trees
for multi-label classification (Esteban et al., *Knowledge-Based Systems* 304,
2024), used unmodified as the referee. Its own licence applies.
