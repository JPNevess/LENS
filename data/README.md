# Streams

Two kinds of stream live here, and only one of them is committed.

## Synthetic: generated, not shipped

The six synthetic streams are defined by `lens/streams.py` and built on demand
the first time an experiment asks for one, then cached in this directory. The
generator and its seeds are what is committed; the files are not, and deleting
one costs only the time to write it again.

| stream | instances | note |
|---|---|---|
| `AGR_a`, `AGR_g` | 100k | Agrawal generator, 5% perturbation, abrupt and gradual drift |
| `LED_a`, `LED_g` | 100k | LED generator, 10% label noise, abrupt and gradual drift |
| `RBF_m`, `RBF_f` | 100k | Radial basis function, moderate and fast incremental drift |

`RBF_a` is also defined — static RBF concepts joined by abrupt drifts — but is
excluded from the studies by default.

To build them ahead of time rather than on first use:

```bash
python experiments/make_datasets.py
python experiments/make_datasets.py --datasets LED_a RBF_m --force
```

The geometry lives in `lens/streams.py`: `STREAM_SEED` fixes the concept
geometry and the instance draw, `GRADUAL_WIDTH` and `GRADUAL_POS` place the
gradual transitions, and `ABRUPT_POS` places the abrupt ones. Changing any of
them changes every stream, so runs made before and after such a change are not
comparable with each other.

## Real: committed or downloaded

| file | instances | note |
|---|---|---|
| `Electricity.arff` | 45k | committed here |
| `airlines.arff` | 539k | committed here |
| `CovtFD.arff` | 581k | built locally, see below |

These cannot be generated from a seed. They have a labelling seed like any other
stream, but the instances themselves are fixed.

### CovtFD is not included

`CovtFD` is a feature-drift variant of Covertype: the 54 real attributes are
kept, 50 noise attributes are appended, and at one third and two thirds of the
stream a block of numeric attributes is swapped with noise. The generated file is
348 MB, above the 100 MB per-file limit of the hosting platform, so it is built
locally:

1. Download the Covertype dataset in ARFF form and save it here as
   `ForestCoverType.arff`.
2. Run:

       python experiments/make_datasets.py --datasets CovtFD

Its seed is `COVTFD_SEED` in `experiments/make_datasets.py`, and it drives only
the noise attributes and the permutation that produces the feature drift — the
real attributes and the labels come from the source file unchanged.

## Note on the committed results

The CSVs under `results/` were produced from an earlier draw of the synthetic
streams, with a different drift geometry and different seeds. Streams generated
by the current code will not reproduce those numbers; see the provenance note in
the top-level `README.md`.
