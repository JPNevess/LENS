"""Materialise the synthetic streams, and build the feature-drift Covertype variant.

The synthetic streams are defined by ``lens/streams.py`` and are generated on
demand the first time an experiment asks for one, so this script is only needed
to build them ahead of time or to rebuild them after changing the geometry
there. CovtFD is the exception: it is derived from the real Covertype dataset
and has to be built explicitly.

    python experiments/make_datasets.py                       # every synthetic stream
    python experiments/make_datasets.py --datasets LED_a RBF_m
    python experiments/make_datasets.py --datasets CovtFD

CovtFD needs data/ForestCoverType.arff as its source; see data/README.md.
"""
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from lens._java import ensure_java_home

ensure_java_home()

import numpy as np
from capymoa.stream import ARFFStream

from lens import streams

OUT_DIR = streams.DATA_DIR

# CovtFD is not seeded from lens.streams: it is a transformation of a real file,
# not a generated stream, so its seed only drives the noise attributes and the
# permutation that produces the feature drift.
COVTFD_SEED = 3607
COVTFD_NOISE_ATTRS = 50
COVTFD_DRIFTING_COLS = 10


def gen_covtfd(out_dir, src=None, n_max=0, seed=COVTFD_SEED):
    """Covertype with noise attributes, two of which swap in for real ones.

    The 54 real attributes are kept, ``COVTFD_NOISE_ATTRS`` noise attributes are
    appended, and at one third and two thirds of the stream a block of numeric
    attributes swaps position with random noise attributes. The label is
    unchanged, so the drift is purely in which columns carry the signal.
    """
    src = src or os.path.join(out_dir, "ForestCoverType.arff")
    if not os.path.exists(src):
        print(f"  [CovtFD] source not found: {src}")
        return None

    schema = ARFFStream(src).get_schema()
    n_real = schema.get_num_attributes()
    n_tot = n_real + COVTFD_NOISE_ATTRS
    numeric_cols = list(range(COVTFD_DRIFTING_COLS))

    print("  [CovtFD] first pass: counting instances and classes")
    s1 = ARFFStream(src)
    ys, n_instances = set(), 0
    while s1.has_more_instances() and (not n_max or n_instances < n_max):
        ys.add(int(s1.next_instance().y_index))
        n_instances += 1
    classes = sorted(ys)
    third = n_instances // 3

    rng_noise = np.random.default_rng(seed)
    rng_perm = np.random.default_rng(seed + 1)

    def layout(segment):
        """Permutation of the sources for one segment of the stream.

        A source below ``n_real`` is a real attribute, the rest are noise. After
        the first segment the drift swaps valid numeric attributes with random
        noise positions.
        """
        base = list(range(n_tot))
        if segment >= 1:
            noise_slots = list(range(n_real, n_tot))
            rng_perm.shuffle(noise_slots)
            for k, col in enumerate(numeric_cols):
                i, j = base.index(col), base.index(noise_slots[k])
                base[i], base[j] = base[j], base[i]
        return base

    layouts = [layout(0), layout(1), layout(2)]

    s2 = ARFFStream(src)
    path = os.path.join(out_dir, "CovtFD.arff")
    tmp = path + ".partial"
    written = 0
    with open(tmp, "w") as f:
        f.write("@relation CovtFD\n\n")
        for i in range(n_tot):
            f.write(f"@attribute att{i} numeric\n")
        f.write("@attribute class {" + ",".join(str(c) for c in classes) + "}\n\n@data\n")
        while s2.has_more_instances() and (not n_max or written < n_max):
            inst = s2.next_instance()
            values = np.concatenate([
                np.asarray(inst.x, dtype=float),
                rng_noise.uniform(0.0, 1.0, COVTFD_NOISE_ATTRS),
            ])
            segment = 0 if written < third else (1 if written < 2 * third else 2)
            f.write(",".join(f"{v:g}" for v in values[layouts[segment]]))
            f.write(f",{int(inst.y_index)}\n")
            written += 1
            if written % 50000 == 0:
                print(f"    ... {written}/{n_instances}")
    os.replace(tmp, path)
    print(f"  -> {path}  ({written} instances, {n_tot} features)")
    return path


def emit_mask_csv(arff_path, label_rate, label_seed):
    """Write a CSV of attributes plus class and a label_revealed mask.

    Only for inspecting a labelling draw; the experiments do not read it. The
    mask has its own seed, independent of the one that generated the stream,
    which is the same split :func:`lens.evaluation.run_experiment` makes.
    """
    import csv

    name = os.path.splitext(os.path.basename(arff_path))[0]
    out = os.path.join(os.path.dirname(arff_path),
                       f"{name}__p{label_rate}_seed{label_seed}.csv")
    stream = ARFFStream(arff_path)
    n_att = stream.get_schema().get_num_attributes()
    rng = np.random.default_rng(label_seed)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"att{i}" for i in range(n_att)]
                        + ["class", "label_revealed"])
        while stream.has_more_instances():
            inst = stream.next_instance()
            writer.writerow(list(inst.x) + [inst.y_label,
                                            bool(rng.random() < label_rate)])
    print(f"  -> {out}  (mask p={label_rate}, seed={label_seed})")


ALL = list(streams.SYNTHETIC) + ["CovtFD"]


def main():
    ap = argparse.ArgumentParser(
        description="Materialise the synthetic streams used in the paper.")
    ap.add_argument("--datasets", nargs="+", default=ALL,
                    help=f"which streams to generate (default: {ALL})")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-stream seed from lens/streams.py; "
                         "leaving this alone is what reproduces the streams the "
                         "committed runs used")
    ap.add_argument("--max-instances", type=int, default=0, dest="n_max",
                    help="cap per stream (0 uses the size from the paper)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if the file is already there")
    ap.add_argument("--label-rate", type=float, default=None,
                    help="also write a CSV with a label_revealed mask at this rate")
    ap.add_argument("--label-seed", type=int, default=4804,
                    help="seed of the labelling mask")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n_instances = args.n_max if args.n_max else streams.N_INSTANCES

    print("=" * 65)
    print(f"  stream seed : {streams.STREAM_SEED if args.seed is None else args.seed}")
    print(f"  instances   : {n_instances}")
    print(f"  gradual     : width {streams.GRADUAL_WIDTH} at {list(streams.GRADUAL_POS)}")
    print(f"  abrupt      : at {list(streams.ABRUPT_POS)}")
    print(f"  output      : {args.out}")
    print("=" * 65)

    for name in args.datasets:
        if name not in ALL:
            print(f"[skip] unknown stream: {name} (available: {ALL})")
            continue

        path = os.path.join(args.out, f"{name}.arff")
        if os.path.exists(path) and not args.force:
            print(f"\n=== {name} already at {path}, pass --force to rebuild ===")
        else:
            print(f"\n=== generating {name} ===")
            if name == "CovtFD":
                path = gen_covtfd(args.out, n_max=args.n_max)
            else:
                path = streams.generate(name, args.out, seed=args.seed,
                                        n_instances=n_instances)

        if args.label_rate is not None and path and os.path.exists(path):
            emit_mask_csv(path, args.label_rate, args.label_seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
