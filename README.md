# Unidirectional Broadcast Coordination

Code accompanying the paper: **"Scale replaces device-level sensing in the dispatch of distributed energy storage"**

## Overview

This repository provides the simulation framework and experiment scripts for the results reported in the paper. The framework models a fleet of distributed battery storage devices that independently respond to a shared broadcast signal, and demonstrates that the aggregate response converges to a deterministic function of the signal parameters as fleet size grows.

## Structure

```
src/
  signal/          64-bit broadcast signal encoding and intensity optimisation
  edge/            Battery model and device state machine
  estimation/      Dual quantile neural network + conformal prediction (CQR)
  simulation/      Agent-based fleet simulator (N heterogeneous batteries)
  analysis/        Bootstrap confidence intervals and statistical tests

experiments/
  run_experiment.py              Unified experiment runner (CLI)

tests/                           Unit and integration tests
data/nextgen/                    Real-parameter validation data (optional)
```

## Requirements

Python >= 3.9 with the following packages:

```
numpy scipy torch matplotlib
```

Install:

```bash
pip install -e .
```

## Reproducing experiments

Each result in the paper maps to a CLI flag:

```bash
# Smoke test (~30 min per result, single run)
python -m experiments.run_experiment --result1 --n-devices 5000 --n-runs 1
python -m experiments.run_experiment --result2 --n-devices 5000
python -m experiments.run_experiment --result3 --n-devices 5000
python -m experiments.run_experiment --result4 --n-devices 5000

# Full 30-run statistical evaluation
python -m experiments.run_experiment --all --n-devices 5000 --n-runs 30
```

| Flag | Figure | Description |
|------|--------|-------------|
| `--result1` | Fig. 2 | 1/√N scaling law, convergence threshold N*, heterogeneity sensitivity |
| `--result2` | Fig. 3 | Broadcast dispatch performance ceiling, curtailment reduction |
| `--result3` | Fig. 4 | Robustness: model mismatch sensitivity, correlation effects |
| `--result4` | Fig. 5 | Generalization: cross-region transfer, real-parameter validation |

Output is saved to `results/experiments/` with timestamped directories.

## Real-parameter validation (optional)

Result 4 includes an optional validation using household battery data from the NextGen project (ACT, Australia). Place the CSV files in `data/nextgen/`. If unavailable, this step is skipped automatically.

**Dataset:** Sturmberg, B. & Shaw, M. (2025). *Select data from the NextGen energy storage trial in the ACT, Australia.* Zenodo. https://doi.org/10.5281/zenodo.14885589

## License

MIT License. See [LICENSE](LICENSE).
