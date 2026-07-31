# Streams

This directory holds the scripts that build the synthetic streams and the
instructions for obtaining the real ones. Generated and downloaded files land
here and are ignored by git.

## Synthetic: one script each

Each script writes one stream to `data/<name>.arff`. The geometry is defined by
`lens/streams.py`, so a stream is fixed by that module plus one integer.

| script | stream | drift |
|---|---|---|
| `make_agr_a.py` | `AGR_a` | Agrawal, 5% perturbation, abrupt change points |
| `make_agr_g.py` | `AGR_g` | Agrawal, gradual transitions |
| `make_led_a.py` | `LED_a` | LED digits, 10% label noise, abrupt change points |
| `make_led_g.py` | `LED_g` | LED digits, gradual transitions |
| `make_rbf_m.py` | `RBF_m` | radial basis, slow continuous centroid drift |
| `make_rbf_f.py` | `RBF_f` | radial basis, fast continuous centroid drift |
| `make_rbf_a.py` | `RBF_a` | static radial basis concepts, abrupt change points |

```bash
python data/make_led_a.py                 # 100k instances, default seed
python data/make_led_a.py --instances 50000
python data/make_led_a.py --seed 1234 --out /tmp/LED_a.arff
```

Building them by hand is optional: an experiment generates the stream it needs
on first use and caches it here.

To build all of them at once:

```bash
python experiments/make_datasets.py
```

### Changing the geometry

`lens/streams.py` holds `STREAM_SEED` for the concept geometry and the instance
draw, `ABRUPT_POS` for the abrupt change points, `GRADUAL_POS` and
`GRADUAL_WIDTH` for the gradual transitions, and `RBF_MODERATE` / `RBF_FAST` for
the two drift speeds. Changing any of them changes every stream built afterwards,
so results produced before and after such a change are not comparable.

## Real: downloaded

These cannot be generated from a seed. Place them in this directory under the
names below.

| file | instances | needed for |
|---|---|---|
| `Electricity.arff` | 45k | the `Electricity` row |
| `airlines.arff` | 539k | the `airlines` row |
| `ForestCoverType.arff` | 581k | building `CovtFD` |

All three are distributed with MOA and are available from the usual
stream-mining dataset collections.

### CovtFD

A feature-drift variant of Covertype: the 54 real attributes are kept, 50 noise
attributes are appended, and at one third and two thirds of the stream a block of
numeric attributes swaps position with noise attributes. The label is unchanged,
so the drift is purely in which columns carry the signal.

```bash
python experiments/make_datasets.py --datasets CovtFD
```

It needs `ForestCoverType.arff` in this directory and produces a 348 MB file.
`COVTFD_SEED` in `experiments/make_datasets.py` drives only the noise attributes
and the permutation; the real attributes and the labels come from the source file
unchanged.
