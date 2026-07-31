"""Generate the RBF_m stream.

Random radial basis functions whose centroids drift slowly and continuously.

Geometry
--------
Five classes, ten numeric attributes, fifty centroids of which ten move. There
is no change point at all: the concept deforms at a constant rate
``lens/streams.RBF_MODERATE`` throughout, which is the regime a change
detector cannot localise.

The stream is defined by ``lens/streams.py`` plus one integer: the seed below.
Change either and you get a different draw, so a set of results is only
comparable with another set generated from the same values.

    python data/make_rbf_m.py                  # writes data/RBF_m.arff
    python data/make_rbf_m.py --instances 50000
    python data/make_rbf_m.py --seed 1234 --out /tmp/RBF_m.arff

Running an experiment generates this stream automatically the first time it is
needed, so this script is only useful to build it ahead of time, to rebuild it
after changing the geometry, or to write a variant somewhere else.
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from lens import streams

NAME = "RBF_m"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--instances", type=int, default=streams.N_INSTANCES,
                   help="how many instances to write")
    p.add_argument("--seed", type=int, default=None,
                   help="override the stream seed (default: the one in lens/streams.py)")
    p.add_argument("--out", default=None,
                   help="output path (default: data/RBF_m.arff)")
    args = p.parse_args(argv)

    path = args.out or os.path.join(streams.DATA_DIR, f"{NAME}.arff")
    print(f"{NAME}: seed {args.seed or streams._seed_for(NAME)}, "
          f"{args.instances} instances")
    streams.write_arff(path, NAME, streams.build(NAME, args.seed), args.instances)
    return path


if __name__ == "__main__":
    main()
