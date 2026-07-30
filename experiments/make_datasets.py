"""Generate the synthetic streams and the feature-drift Covertype variant.

The six synthetic streams are produced from generators with fixed seeds. CovtFD
is derived from the Covertype dataset: the 54 real attributes are kept, 50 noise
attributes are appended, and at one third and two thirds of the stream a block of
attributes is swapped with noise, which is the feature drift.

    python experiments/make_datasets.py --datasets LED_a RBF_m
    python experiments/make_datasets.py --datasets CovtFD

CovtFD needs data/ForestCoverType.arff as its source; see data/README.md.
"""
import os
import sys
import argparse

if not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"

import numpy as np
from capymoa.stream.generator import (
    LEDGeneratorDrift, AgrawalGenerator, RandomRBFGeneratorDrift,
)
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream import ARFFStream

_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "data")
GRADUAL_WIDTH = 1000

ABRUPT_POS   = lambda n, k: [n * (i + 1) // (k + 1) for i in range(k)]
GRADUAL_POS  = [20000, 28000, 43000, 50000, 70000, 72000, 79000]

def _drift_stream(concepts, positions, gradual, seed):
    parts = [concepts[0]]
    for i, pos in enumerate(positions):
        if gradual:
            parts.append(GradualDrift(position=int(pos), width=GRADUAL_WIDTH, random_seed=seed))
        else:
            parts.append(AbruptDrift(position=int(pos), random_seed=seed))
        parts.append(concepts[i + 1])
    return DriftStream(stream=parts)


def make_led(n, gradual, seed):
    positions = GRADUAL_POS if gradual else ABRUPT_POS(n, 4)
    drift_attrs = [0, 1, 2, 3, 4, 5, 6, 7]
    concepts = [
        LEDGeneratorDrift(instance_random_seed=seed + i, noise_percentage=10,
                          number_of_attributes_with_drift=drift_attrs[i % len(drift_attrs)])
        for i in range(len(positions) + 1)
    ]
    return _drift_stream(concepts, positions, gradual, seed)


def make_agr(n, gradual, seed):
    positions = GRADUAL_POS if gradual else ABRUPT_POS(n, 4)
    funcs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    concepts = [
        AgrawalGenerator(instance_random_seed=seed + i,
                         classification_function=funcs[i % len(funcs)], peturbation=0.05)
        for i in range(len(positions) + 1)
    ]
    return _drift_stream(concepts, positions, gradual, seed)


def make_rbf(speed, seed):
    return RandomRBFGeneratorDrift(
        model_random_seed=seed, instance_random_seed=seed + 1,
        number_of_classes=5, number_of_attributes=10, number_of_centroids=50,
        number_of_drifting_centroids=10, magnitude_of_change=speed,
    )


def make_rbf_abrupt(n, seed):
    """Static RandomRBF concepts with different seeds, joined by abrupt drifts.

    Each concept has a fresh centroid geometry and the concepts are joined at
    20k/40k/60k/80k, so every jump is an instantaneous change of concept.
    """
    positions = ABRUPT_POS(n, 4)
    concepts = [
        RandomRBFGeneratorDrift(
            model_random_seed=seed + 17 * i, instance_random_seed=seed + 17 * i + 1,
            number_of_classes=5, number_of_attributes=10, number_of_centroids=50,
            number_of_drifting_centroids=0, magnitude_of_change=0.0,
        )
        for i in range(len(positions) + 1)
    ]
    return _drift_stream(concepts, positions, gradual=False, seed=seed)


def _write_arff(path, name, n_att, class_labels, instance_iter, n_total):
    with open(path, "w") as f:
        f.write(f"@relation {name}\n\n")
        for i in range(n_att):
            f.write(f"@attribute att{i} numeric\n")
        f.write("@attribute class {" + ",".join(str(c) for c in class_labels) + "}\n\n@data\n")
        for k, (x, y) in enumerate(instance_iter):
            f.write(",".join(f"{v:g}" for v in x) + f",{y}\n")
            if (k + 1) % 50000 == 0:
                print(f"    ... {k+1}/{n_total}")
    print(f"  -> {path}  ({n_total} instances, {n_att} features)")


def _stream_iter(stream, n):
    c = 0
    while stream.has_more_instances() and c < n:
        inst = stream.next_instance()
        yield np.asarray(inst.x, dtype=float), inst.y_label
        c += 1


def gen_synthetic(name, stream, n, out_dir):
    sch = stream.get_schema()
    n_att = sch.get_num_attributes()
    labels = sch.get_label_values()
    _write_arff(os.path.join(out_dir, f"{name}.arff"), name, n_att, labels,
                _stream_iter(stream, n), n)


def gen_covtfd(out_dir, src=None, n_max=0, seed=42):
    src = src or os.path.join(_HERE, "data", "ForestCoverType.arff")
    if not os.path.exists(src):
        print(f"  [CovtFD] source not found: {src}")
        return
    sch = ARFFStream(src).get_schema()
    n_real  = sch.get_num_attributes()
    n_noise = 50
    n_tot   = n_real + n_noise
    numeric_cols = list(range(10))

    print("  [CovtFD] first pass: counting instances and classes")
    s1 = ARFFStream(src)
    ys, N = set(), 0
    while s1.has_more_instances() and (not n_max or N < n_max):
        ys.add(int(s1.next_instance().y_index)); N += 1
    classes = sorted(ys)
    third = N // 3

    rng_noise = np.random.default_rng(seed)
    rng_perm  = np.random.default_rng(seed + 1)

    def layout(seg):
        """Permutation of the 104 sources for one segment.

        A source below 54 is a real attribute, the rest are noise. The drift swaps
        the positions of valid numeric attributes with random noise positions.
        """
        base = list(range(n_tot))
        if seg >= 1:
            noise_slots = list(range(n_real, n_tot))
            rng_perm.shuffle(noise_slots)
            for k, nc in enumerate(numeric_cols):
                ci, cj = base.index(nc), base.index(noise_slots[k])
                base[ci], base[cj] = base[cj], base[ci]
        return base
    layouts = [layout(0), layout(1), layout(2)]

    s2 = ARFFStream(src)
    def it():
        c = 0
        while s2.has_more_instances() and (not n_max or c < n_max):
            inst = s2.next_instance()
            src_vals = np.concatenate([np.asarray(inst.x, dtype=float),
                                       rng_noise.uniform(0.0, 1.0, n_noise)])
            seg = 0 if c < third else (1 if c < 2 * third else 2)
            yield src_vals[layouts[seg]], int(inst.y_index)
            c += 1

    _write_arff(os.path.join(out_dir, "CovtFD.arff"), "CovtFD", n_tot, classes, it(), N)


def emit_mask_csv(arff_path, label_rate, label_seed):
    """Write a CSV of attributes plus class and a label_revealed mask.

    The mask is drawn from its own seed, independent of the one used to
    generate the stream.
    """
    import csv
    name = os.path.splitext(os.path.basename(arff_path))[0]
    out = os.path.join(os.path.dirname(arff_path), f"{name}__p{label_rate}_seed{label_seed}.csv")
    stream = ARFFStream(arff_path)
    sch = stream.get_schema()
    n_att = sch.get_num_attributes()
    rng = np.random.default_rng(label_seed)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"att{i}" for i in range(n_att)] + ["class", "label_revealed"])
        while stream.has_more_instances():
            inst = stream.next_instance()
            revealed = bool(rng.random() < label_rate)
            w.writerow(list(inst.x) + [inst.y_label, revealed])
    print(f"  -> {out}  (mask p={label_rate}, seed={label_seed})")


ALL = ["LED_a", "LED_g", "AGR_a", "AGR_g", "RBF_m", "RBF_f", "RBF_a", "CovtFD"]


def main():
    ap = argparse.ArgumentParser(
        description="Generate the synthetic streams used in the paper.")
    ap.add_argument("--datasets", nargs="+", default=ALL, help=f"quais gerar (default: {ALL})")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42, help="seed of the data and drift generators")
    ap.add_argument("--max-instances", type=int, default=0, dest="n_max",
                    help="cap per stream (0 uses the size from the paper)")
    ap.add_argument("--label-rate", type=float, default=None,
                    help="also write a CSV with a label_revealed mask at this rate")
    ap.add_argument("--label-seed", type=int, default=1,
                    help="seed of the labelling mask")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n100 = args.n_max if args.n_max else 100000
    builders = {
        "LED_a": lambda: gen_synthetic("LED_a", make_led(n100, False, args.seed), n100, args.out),
        "LED_g": lambda: gen_synthetic("LED_g", make_led(n100, True,  args.seed), n100, args.out),
        "AGR_a": lambda: gen_synthetic("AGR_a", make_agr(n100, False, args.seed), n100, args.out),
        "AGR_g": lambda: gen_synthetic("AGR_g", make_agr(n100, True,  args.seed), n100, args.out),
        "RBF_m": lambda: gen_synthetic("RBF_m", make_rbf(0.0001, args.seed), n100, args.out),
        "RBF_f": lambda: gen_synthetic("RBF_f", make_rbf(0.001,  args.seed), n100, args.out),
        "RBF_a": lambda: gen_synthetic("RBF_a", make_rbf_abrupt(n100, args.seed), n100, args.out),
        "CovtFD": lambda: gen_covtfd(args.out, n_max=args.n_max, seed=args.seed),
    }

    for ds in args.datasets:
        if ds not in builders:
            print(f"[skip] unknown stream: {ds} (available: {ALL})"); continue
        print(f"\n=== A gerar {ds} ===")
        builders[ds]()
        arff = os.path.join(args.out, f"{ds}.arff")
        if args.label_rate is not None and os.path.exists(arff):
            emit_mask_csv(arff, args.label_rate, args.label_seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
