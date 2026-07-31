"""Generate the LED_g stream.

The same LED digits, but the concept changes are gradual rather than
instantaneous.

Geometry
--------
Identical to LED_a except that the transitions are spread over
``GRADUAL_WIDTH`` instances, during which the two concepts are mixed. The
centres are ``lens/streams.GRADUAL_POS``, which are clustered in places and
isolated in others, so the stream has both rapid successions of change and
long stable stretches.

The stream is defined by ``lens/streams.py`` plus one integer: the seed below.
Change either and you get a different draw, so a set of results is only
comparable with another set generated from the same values.

    python data/make_led_g.py                  # writes data/LED_g.arff
    python data/make_led_g.py --instances 50000
    python data/make_led_g.py --seed 1234 --out /tmp/LED_g.arff

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

NAME = "LED_g"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--instances", type=int, default=streams.N_INSTANCES,
                   help="how many instances to write")
    p.add_argument("--seed", type=int, default=None,
                   help="override the stream seed (default: the one in lens/streams.py)")
    p.add_argument("--out", default=None,
                   help="output path (default: data/LED_g.arff)")
    args = p.parse_args(argv)

    path = args.out or os.path.join(streams.DATA_DIR, f"{NAME}.arff")
    print(f"{NAME}: seed {args.seed or streams._seed_for(NAME)}, "
          f"{args.instances} instances")
    streams.write_arff(path, NAME, streams.build(NAME, args.seed), args.instances)
    return path


if __name__ == "__main__":
    main()
