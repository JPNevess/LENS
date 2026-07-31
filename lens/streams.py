"""Construction of the synthetic streams, and resolution of a stream name to data.

The synthetic streams are not shipped as files. They are built here from CapyMOA
generators, so the stream a run saw is defined by this module plus one integer
and nothing else: change ``STREAM_SEED`` or any constant below and you get a
different draw. The real streams cannot be generated and are read from ``data/``.

Two independent seeds decide what a run sees, and they are deliberately kept
apart:

``STREAM_SEED``
    the concept geometry, the drift transitions and the instance draw -- what
    the stream *is*. Only synthetic streams have one.

``label_seed`` (passed to :func:`lens.evaluation.run_experiment`)
    which instances arrive with their label revealed -- what the learner is
    *allowed to see*. Both synthetic and real streams have one, so the labelling
    protocol is the same either way.

A synthetic stream is materialised to ``data/<name>.arff`` the first time it is
needed and reused afterwards; the file is a cache, not a source, and deleting it
only costs the time to write it again.

"""
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

# --------------------------------------------------------------------- geometry
# Seed of every synthetic stream. One integer fixes the concept geometry, the
# drift transitions and the instance draw for all six of them; the per-stream
# offsets below keep the families independent of each other.
STREAM_SEED = 100

_OFFSET = {
    "LED_a": 0, "LED_g": 1, "AGR_a": 2, "AGR_g": 3,
    "RBF_m": 4, "RBF_f": 5, "RBF_a": 6,
}

N_INSTANCES = 100_000

# Width of a gradual transition, in instances. The two concepts are mixed across
# this window, so a larger value is a slower, harder-to-localise drift.
GRADUAL_WIDTH = 3000

# Abrupt change points. Irregularly spaced, so that a detector cannot do well by
# assuming a fixed period between drifts.
ABRUPT_POS = (18_500, 37_000, 58_000, 81_500)

# Centres of the gradual transitions. Clustered in places, isolated in others, so
# the stream contains both a rapid succession of changes and long stable stretches.
GRADUAL_POS = (15_000, 26_500, 38_000, 52_500, 64_000, 77_500, 88_000)

# Speed of the incremental centroid drift in the two RandomRBF streams.
RBF_MODERATE = 0.00012
RBF_FAST = 0.0012

SYNTHETIC = ("LED_a", "LED_g", "AGR_a", "AGR_g", "RBF_m", "RBF_f", "RBF_a")
REAL = ("airlines", "Electricity", "CovtFD", "ForestCoverType")

# The streams every study runs over by default. Discovering them by globbing
# ``data/`` would silently shrink a study to whatever files happen to be on disk,
# and the synthetic ones are generated rather than shipped, so the list is
# explicit instead.
DEFAULT = ("LED_a", "LED_g", "AGR_a", "AGR_g", "RBF_m", "RBF_f",
           "airlines", "Electricity", "CovtFD")

# Excluded from the default sweep of the heavier studies: hours per run.
BIG = ("CovtFD", "ForestCoverType", "PokerHand")


def _seed_for(name):
    return STREAM_SEED + _OFFSET.get(name, 0)


# -------------------------------------------------------------------- builders
def _drift_stream(concepts, positions, gradual, seed):
    """Chain concepts together, separated by abrupt or gradual transitions."""
    from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift

    parts = [concepts[0]]
    for i, pos in enumerate(positions):
        if gradual:
            parts.append(GradualDrift(position=int(pos), width=GRADUAL_WIDTH,
                                      random_seed=seed))
        else:
            parts.append(AbruptDrift(position=int(pos), random_seed=seed))
        parts.append(concepts[i + 1])
    return DriftStream(stream=parts)


def make_led(gradual, seed):
    """LED with 10% label noise; each concept drifts a different attribute."""
    from capymoa.stream.generator import LEDGeneratorDrift

    positions = GRADUAL_POS if gradual else ABRUPT_POS
    drift_attrs = [3, 0, 6, 1, 5, 2, 7, 4]
    concepts = [
        LEDGeneratorDrift(
            instance_random_seed=seed + 31 * i, noise_percentage=10,
            number_of_attributes_with_drift=drift_attrs[i % len(drift_attrs)])
        for i in range(len(positions) + 1)
    ]
    return _drift_stream(concepts, positions, gradual, seed)


def make_agr(gradual, seed):
    """Agrawal with 5% perturbation; each concept uses a different rule."""
    from capymoa.stream.generator import AgrawalGenerator

    positions = GRADUAL_POS if gradual else ABRUPT_POS
    funcs = [2, 7, 4, 9, 1, 6, 3, 10, 5, 8]
    concepts = [
        AgrawalGenerator(instance_random_seed=seed + 31 * i,
                         classification_function=funcs[i % len(funcs)],
                         peturbation=0.05)
        for i in range(len(positions) + 1)
    ]
    return _drift_stream(concepts, positions, gradual, seed)


def make_rbf(speed, seed):
    """RandomRBF whose centroids drift continuously; no discrete change point."""
    from capymoa.stream.generator import RandomRBFGeneratorDrift

    return RandomRBFGeneratorDrift(
        model_random_seed=seed, instance_random_seed=seed + 1,
        number_of_classes=5, number_of_attributes=10, number_of_centroids=50,
        number_of_drifting_centroids=10, magnitude_of_change=speed,
    )


def make_rbf_abrupt(seed):
    """Static RandomRBF concepts with fresh geometry, joined by abrupt drifts."""
    from capymoa.stream.generator import RandomRBFGeneratorDrift

    concepts = [
        RandomRBFGeneratorDrift(
            model_random_seed=seed + 31 * i, instance_random_seed=seed + 31 * i + 1,
            number_of_classes=5, number_of_attributes=10, number_of_centroids=50,
            number_of_drifting_centroids=0, magnitude_of_change=0.0,
        )
        for i in range(len(ABRUPT_POS) + 1)
    ]
    return _drift_stream(concepts, ABRUPT_POS, gradual=False, seed=seed)


def build(name, seed=None):
    """Build the CapyMOA stream for one synthetic stream name."""
    if name not in SYNTHETIC:
        raise ValueError(f"{name!r} is not a synthetic stream; "
                         f"available: {list(SYNTHETIC)}")
    seed = _seed_for(name) if seed is None else seed
    builders = {
        "LED_a": lambda: make_led(False, seed),
        "LED_g": lambda: make_led(True, seed),
        "AGR_a": lambda: make_agr(False, seed),
        "AGR_g": lambda: make_agr(True, seed),
        "RBF_m": lambda: make_rbf(RBF_MODERATE, seed),
        "RBF_f": lambda: make_rbf(RBF_FAST, seed),
        "RBF_a": lambda: make_rbf_abrupt(seed),
    }
    return builders[name]()


# ------------------------------------------------------------------ file output
def write_arff(path, name, stream, n_instances=N_INSTANCES, progress=True):
    """Materialise ``n_instances`` of a stream to an ARFF file."""
    schema = stream.get_schema()
    n_att = schema.get_num_attributes()
    labels = schema.get_label_values()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".partial"
    written = 0
    with open(tmp, "w") as f:
        f.write(f"@relation {name}\n\n")
        for i in range(n_att):
            f.write(f"@attribute att{i} numeric\n")
        f.write("@attribute class {" + ",".join(str(c) for c in labels) + "}\n\n@data\n")
        while stream.has_more_instances() and written < n_instances:
            inst = stream.next_instance()
            f.write(",".join(f"{v:g}" for v in np.asarray(inst.x, dtype=float)))
            f.write(f",{inst.y_label}\n")
            written += 1
            if progress and written % 25000 == 0:
                print(f"    ... {written}/{n_instances}", flush=True)
    # Rename only once the file is complete, so an interrupted run does not leave
    # a truncated stream behind that later looks like a valid cache.
    os.replace(tmp, path)
    if progress:
        print(f"  -> {path}  ({written} instances, {n_att} features)")
    return path


def generate(name, out_dir=DATA_DIR, seed=None, n_instances=N_INSTANCES,
             progress=True):
    """Build one synthetic stream and write it to ``out_dir/<name>.arff``."""
    path = os.path.join(out_dir, f"{name}.arff")
    return write_arff(path, name, build(name, seed), n_instances, progress)


# -------------------------------------------------------------------- resolution
def resolve(name, data_dir=DATA_DIR, n_instances=N_INSTANCES, progress=True):
    """Return a path to the stream, generating it first if it is synthetic.

    Synthetic streams are generated on demand and cached under ``data_dir``. Real
    streams have to be present already: they cannot be produced from a seed.
    """
    for ext in (".arff", ".csv"):
        path = os.path.join(data_dir, name + ext)
        if os.path.exists(path):
            return path

    if name in SYNTHETIC:
        if progress:
            print(f"  [streams] generating {name} "
                  f"(seed {_seed_for(name)}, {n_instances} instances); "
                  f"this is cached in {data_dir}", flush=True)
        return generate(name, data_dir, n_instances=n_instances, progress=progress)

    raise FileNotFoundError(
        f"stream {name!r} not found in {data_dir}. It is a real stream, so it "
        f"cannot be generated from a seed; see data/README.md for where to get "
        f"it. Synthetic streams ({', '.join(SYNTHETIC)}) are generated "
        f"automatically.")


def resolve_all(names=None, data_dir=DATA_DIR, include_big=True, progress=True):
    """Resolve several stream names to paths, generating the synthetic ones.

    A real stream that is not on disk is reported and skipped rather than raised
    on, so a study still runs over the streams that are available. A synthetic
    one is always generated, so it can never be skipped by accident.
    """
    names = list(DEFAULT if names is None else names)
    if not include_big:
        skipped = [n for n in names if n in BIG]
        if skipped and progress:
            print(f"  [skip] large streams excluded by default: {skipped} "
                  f"(pass --include-big or name them explicitly to run them)")
        names = [n for n in names if n not in BIG]

    paths = []
    for name in names:
        try:
            paths.append(resolve(name, data_dir, progress=progress))
        except FileNotFoundError as exc:
            if progress:
                print(f"  [skip] {exc}")
    return paths
