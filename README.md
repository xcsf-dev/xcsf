# XCSF learning classifier system

[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0)
[![Linux Build](https://img.shields.io/github/actions/workflow/status/xcsf-dev/xcsf/ubuntu_build.yml?branch=master&logo=linux&logoColor=white&style=flat&label=Ubuntu)](https://github.com/xcsf-dev/xcsf/actions?query=workflow%3A%22Ubuntu+build%22)
[![MacOS Build](https://img.shields.io/github/actions/workflow/status/xcsf-dev/xcsf/macOS_build.yml?branch=master&logo=apple&logoColor=white&style=flat&label=macOS)](https://github.com/xcsf-dev/xcsf/actions?query=workflow%3A%22macOS+build%22)
[![Windows Build](https://img.shields.io/github/actions/workflow/status/xcsf-dev/xcsf/windows_build.yml?branch=master&logo=windows&logoColor=white&style=flat&label=Windows)](https://github.com/xcsf-dev/xcsf/actions?query=workflow%3A%22Windows+build%22)
[![Latest Version](https://img.shields.io/github/v/release/xcsf-dev/xcsf?style=flat)](https://github.com/xcsf-dev/xcsf/releases)
[![DOI](https://zenodo.org/badge/28035841.svg)](https://zenodo.org/badge/latestdoi/28035841)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/xcsf-dev/xcsf?logo=codefactor&style=flat)](https://www.codefactor.io/repository/github/xcsf-dev/xcsf)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=xcsf-dev_xcsf&metric=alert_status)](https://sonarcloud.io/dashboard?id=xcsf-dev_xcsf)
[![codecov](https://codecov.io/gh/xcsf-dev/xcsf/branch/master/graph/badge.svg?token=3bfaTvmJ8d)](https://codecov.io/gh/xcsf-dev/xcsf)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=xcsf-dev_xcsf&metric=ncloc)](https://sonarcloud.io/dashboard?id=xcsf-dev_xcsf)
[![PyPI package](https://img.shields.io/pypi/v/xcsf.svg)](https://pypi.org/project/xcsf)
[![Python versions](https://img.shields.io/pypi/pyversions/xcsf.svg)](https://pypi.org/project/xcsf)
[![Downloads](https://static.pepy.tech/personalized-badge/xcsf?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/xcsf)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/xcsf-dev/xcsf)

XCSF is an accuracy-based [learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) for [online](https://en.wikipedia.org/wiki/Online_machine_learning) [supervised](https://en.wikipedia.org/wiki/Supervised_learning) and [reinforcement](https://en.wikipedia.org/wiki/Reinforcement_learning) learning. Rather than fitting a single global model, it uses an [evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_computation) to design a population of [rules](https://en.wikipedia.org/wiki/Rule-based_machine_learning), each of which matches a region of the input space and predicts with its own simple model (for example, a [linear regression](https://en.wikipedia.org/wiki/Linear_regression) or a small [neural network](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))).

This repository provides a C implementation that can be built as a stand-alone binary or used as a Python library with a [scikit-learn](https://scikit-learn.org)-style interface. See the project [wiki](https://github.com/xcsf-dev/xcsf/wiki) for features, build instructions, and usage.

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
    return np.sin(2 * np.pi * x)  # target function


rng = np.random.default_rng(1)
X_train = rng.random((1000, 1))
X_test = np.linspace(0, 1, 200).reshape(-1, 1)

xcs = xcsf.XCS(
    x_dim=1,
    y_dim=1,
    n_actions=1,  # 1 = supervised learning
    pop_size=500,
    max_trials=20000,
    random_state=1,
    loss_func="mse",
    e0=0.0001,  # target MSE
    alpha=1,  # recommended for regression
    prediction={"type": "rls_linear"},  # local linear models
)

xcs.fit(X_train, f(X_train), verbose=False)
print(f"Test MSE: {xcs.score(X_test, f(X_test)):.1e}")  # ~3.1e-05
```

For more Python examples, see the [`python`](https://github.com/xcsf-dev/xcsf/tree/master/python) folder, which contains scripts and the equivalent [Jupyter notebooks](https://github.com/xcsf-dev/xcsf/tree/master/python/notebooks).

## Contributing

Contributions to this repository are very welcome.

See our [contributing](https://github.com/xcsf-dev/xcsf/wiki/Contributing) guide for details.
