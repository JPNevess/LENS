# Streams

The six synthetic streams and the two small real ones are included here. They are
what the figures and tables in `results/` were produced from.

| file | instances | note |
|---|---|---|
| `AGR_a.arff`, `AGR_g.arff` | 100k | Agrawal generator, 5% noise, abrupt and gradual drift |
| `LED_a.arff`, `LED_g.arff` | 100k | LED generator, 10% label noise, abrupt and gradual drift |
| `RBF_m.arff`, `RBF_f.arff` | 100k | Radial basis function, moderate and fast incremental drift |
| `Electricity.arff` | 45k | real |
| `airlines.arff` | 539k | real |

## CovtFD is not included

`CovtFD` is a feature-drift variant of Covertype: the 54 real attributes are kept,
50 noise attributes are appended, and at one third and two thirds of the stream a
block of numeric attributes is swapped with noise. The generated file is 348 MB,
which is above the 100 MB per-file limit of the hosting platform, so it is built
locally instead:

1. Download the Covertype dataset in ARFF form and save it here as
   `ForestCoverType.arff`.
2. Run:

       python experiments/make_datasets.py --datasets CovtFD

The synthetic streams can be regenerated the same way, for example:

    python experiments/make_datasets.py --datasets LED_a LED_g RBF_m RBF_f
