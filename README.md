# XCSF learning classifier system

[![Latest Version](https://img.shields.io/github/v/release/xcsf-dev/xcsf?style=flat)](https://github.com/xcsf-dev/xcsf/releases)
[![PyPI package](https://img.shields.io/pypi/v/xcsf.svg)](https://pypi.org/project/xcsf)
[![Linux Build](https://img.shields.io/github/actions/workflow/status/xcsf-dev/xcsf/ubuntu_build.yml?branch=master&logo=linux&logoColor=white&style=flat&label=Ubuntu)](https://github.com/xcsf-dev/xcsf/actions?query=workflow%3A%22Ubuntu+build%22)
[![MacOS Build](https://img.shields.io/github/actions/workflow/status/xcsf-dev/xcsf/macOS_build.yml?branch=master&logo=apple&logoColor=white&style=flat&label=macOS)](https://github.com/xcsf-dev/xcsf/actions?query=workflow%3A%22macOS+build%22)
[![Windows Build](https://img.shields.io/github/actions/workflow/status/xcsf-dev/xcsf/windows_build.yml?branch=master&logo=windows&logoColor=white&style=flat&label=Windows)](https://github.com/xcsf-dev/xcsf/actions?query=workflow%3A%22Windows+build%22)
[![DOI](https://zenodo.org/badge/28035841.svg)](https://zenodo.org/badge/latestdoi/28035841)

XCSF is an accuracy-based [learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) for online supervised and reinforcement learning. Rather than fitting a single global model, it uses an evolutionary algorithm to design a population of rules, each of which matches a region of the input space and predicts with its own simple model (for example, a linear regression or a small neural network).

This repository provides a C implementation that can be built as a stand-alone binary or used as a Python library with a scikit-learn-style interface. See the project [wiki](https://github.com/xcsf-dev/xcsf/wiki) for features, build instructions, and usage.

## Quick start

### Installation

```shell
pip install xcsf
```

### Minimal example

Fit y = sin(2πx) using local linear models:

```python
import numpy as np
import xcsf


def f(x):
    return np.sin(2 * np.pi * x)


rng = np.random.default_rng(1)
X_train = rng.random((1000, 1))
X_test = np.linspace(0, 1, 200).reshape(-1, 1)

xcs = xcsf.XCS(
    x_dim=1,
    y_dim=1,
    n_actions=1,
    pop_size=500,
    max_trials=20000,
    random_state=1,
    loss_func="mse",
    e0=0.0001,
    alpha=1,
    prediction={"type": "rls_linear"},
)

xcs.fit(X_train, f(X_train), verbose=False)
print(f"Test MSE: {xcs.score(X_test, f(X_test)):.1e}")  # ~3.1e-05
```

For more Python examples, see the [`python`](https://github.com/xcsf-dev/xcsf/tree/master/python) folder, which contains scripts and the equivalent [notebooks](https://github.com/xcsf-dev/xcsf/tree/master/python/notebooks).

## Contributing

Contributions to this repository are very welcome.

See our [contributing](https://github.com/xcsf-dev/xcsf/wiki/Contributing) guide for details.
