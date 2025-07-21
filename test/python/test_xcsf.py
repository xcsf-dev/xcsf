#!/usr/bin/python3
#
# Copyright (C) 2024 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""XCSF Python tests."""

from __future__ import annotations

import json
import numbers
import os
import pickle
from collections import namedtuple
from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import xcsf

SEED: int = 101
PKL_FILENAME: str = "blah.pkl"
POP_FILENAME: str = "pset.json"

Data = namedtuple(
    "Data",
    ["x_dim", "y_dim", "x_train", "y_train", "x_val", "y_val", "x_test", "y_test"],
)

np.random.seed(SEED)


@pytest.fixture(scope="module")
def data() -> Data:
    """Load test regression data."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=5,
        n_targets=1,
        random_state=SEED,
    )

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X = feature_scaler.fit_transform(X)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    output_scaler = MinMaxScaler(feature_range=(0, 1))
    y = output_scaler.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=SEED
    )

    return Data(
        np.shape(X)[1],
        np.shape(y)[1],
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


def dicts_equal(d1: dict, d2: dict) -> bool:
    """Return whether all items in d1 are present and equal in d2.

    Recursively checks if all items in the first dictionary (d1) are present and
    equal to the corresponding items in the second dictionary (d2). This function
    also handles nested dictionaries.
    """
    for key, value in d1.items():
        if key not in d2:
            return False
        if isinstance(value, dict):
            if not isinstance(d2[key], dict) or not dicts_equal(value, d2[key]):
                return False
        elif d2[key] != value:
            return False
    return True


def predictions() -> list[dict]:
    """Return list of prediction args."""
    return [
        {"type": "constant"},
        {"type": "nlms_linear"},
        {"type": "nlms_quadratic"},
        {"type": "rls_linear"},
        {"type": "rls_quadratic"},
        {"type": "neural"},
    ]


def conditions() -> list[dict]:
    """Return list of condition args."""
    return [
        {"type": "dummy"},
        {"type": "ternary"},
        {"type": "hyperrectangle_ubr"},
        {"type": "hyperrectangle_csr"},
        {"type": "hyperellipsoid"},
        {"type": "neural"},
        {"type": "tree_gp"},
        {"type": "dgp"},
    ]


@pytest.mark.parametrize("prediction", predictions())
def test_deterministic_prediction(data, prediction):
    """Test deterministic prediction."""
    # create model
    xcs = xcsf.XCS(
        x_dim=data.x_dim,
        y_dim=data.y_dim,
        n_actions=1,
        pop_size=200,
        max_trials=1000,
        random_state=SEED,
        prediction=prediction,
    )

    # fit model
    xcs.fit(data.x_train, data.y_train, validation_data=(data.x_val, data.y_val))

    # get predictions
    a: np.ndarray = xcs.predict(data.x_test)

    # compare output shape
    assert a.shape == data.y_test.shape

    # compare subsequent calls to predict
    b: np.ndarray = xcs.predict(data.x_test)
    assert np.all(a == b)


@pytest.mark.parametrize("condition", conditions())
@pytest.mark.parametrize("prediction", predictions())
def test_serialization(data, condition, prediction):
    """Test saving and loading.

    Note:
    -----
    Calling `predict()` will modify internal parameters such as the current
    prediction, matching state, etc., so we have make sure to perform
    comparisons before modification.
    """
    # create model
    xcs1 = xcsf.XCS(
        x_dim=data.x_dim,
        y_dim=data.y_dim,
        n_actions=1,
        pop_size=20,
        max_trials=100,
        random_state=SEED,
        condition=condition,
        prediction=prediction,
    )

    # fit model
    xcs1.fit(data.x_train, data.y_train, validation_data=(data.x_val, data.y_val))

    # save with pickle
    with open(PKL_FILENAME, "wb") as fp:
        pickle.dump(xcs1, fp)

    # load from pickle
    with open(PKL_FILENAME, "rb") as fp:
        xcs2 = pickle.load(fp)

    # compare loaded instance
    assert isinstance(xcs2, xcsf.XCS)

    # compare parameters
    orig_params: dict = xcs1.internal_params()
    new_params: dict = xcs2.internal_params()
    assert orig_params == new_params

    # compare populations
    orig_pop: str = xcs1.json()
    new_pop: str = xcs2.json()
    assert orig_pop == new_pop

    # compare predictions
    orig_pred: np.ndarray = xcs1.predict(data.x_test)
    new_pred: np.ndarray = xcs2.predict(data.x_test)
    assert np.all(orig_pred == new_pred)

    # clean up
    if os.path.exists(PKL_FILENAME):
        os.remove(PKL_FILENAME)


def test_seeding(data):
    """Test population seeding.

    Currently only tests hyperrectangle ubr.
    """
    # create human-designed classifier
    classifier: dict = {
        "error": 0.05,
        "fitness": 0.3,
        "set_size": 100,
        "numerosity": 2,
        "experience": 3,
        "time": 3,
        "samples_seen": 2,
        "samples_matched": 1,
        "condition": {
            "type": "hyperrectangle_ubr",
            "bound1": np.round(np.random.rand(data.x_dim), 10).tolist(),
            "bound2": np.round(np.random.rand(data.x_dim), 10).tolist(),
            "mutation": [0.2],
        },
    }

    # write population set file
    with open(POP_FILENAME, "w", encoding="utf-8") as file:
        pset = {"classifiers": [classifier]}
        json.dump(pset, file)

    # create model with initial population set from a file
    xcs = xcsf.XCS(
        x_dim=data.x_dim,
        y_dim=data.y_dim,
        n_actions=1,
        random_state=SEED,
        population_file=POP_FILENAME,
        max_trials=1,
        pop_init=False,
        action={"type": "integer"},
        condition={"type": "hyperrectangle_ubr"},
        prediction={"type": "nlms_linear"},
    )

    # add human-designed classifier again, this time manually
    clj: str = json.dumps(classifier)  # dictionary to JSON
    xcs.json_insert_cl(clj)

    # get current population
    pop: list[dict] = json.loads(xcs.json())["classifiers"]

    # check two classifiers are present
    assert len(pop) == 2

    # check population has been seeded correctly
    for cl in pop:
        assert dicts_equal(classifier, cl)

    # fit a single sample
    X1 = data.x_train[0].reshape(1, -1)
    y1 = data.y_train[0].reshape(1, -1)
    xcs.fit(X1, y1, warm_start=True)

    # get updated population
    new_pop: list[dict] = json.loads(xcs.json())["classifiers"]

    # check an additional classifier has been added via covering
    assert len(new_pop) == 3

    # check conditions are still present
    classifier = {k: classifier[k] for k in ["condition"] if k in classifier}
    for cl in new_pop[1:]:  # skip first (covered) classifier
        assert dicts_equal(classifier, cl)

    # clean up
    if os.path.exists(POP_FILENAME):
        os.remove(POP_FILENAME)


def _compare_dicts(d1, d2, path=""):
    diffs = []
    all_keys = set(d1.keys()) | set(d2.keys())

    for key in all_keys:
        subpath = f"{path}.{key}" if path else key

        if key not in d1 or key not in d2:
            diffs.append((subpath, "Path exists in only one dict"))
            continue

        v1, v2 = d1[key], d2[key]

        if isinstance(v1, dict) and isinstance(v2, dict):
            diffs.extend(_compare_dicts(v1, v2, subpath))
        elif isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                diffs.append((subpath, f"List length differs: {len(v1)} != {len(v2)}"))
            for i, (x, y) in enumerate(zip(v1, v2)):
                diffs.extend(_compare_dicts({0: x}, {0: y}, f"{subpath}[{i}]"))
        elif isinstance(v1, numbers.Real) and isinstance(v2, numbers.Real):
            if not np.isclose(v1, v2, atol=1e-10, rtol=0.0):
                diffs.append((subpath, f"{v1} != {v2}"))
        elif v1 != v2:
            diffs.append((subpath, f"{v1} != {v2}"))

    return diffs


def _test_pop_replace(
    tmp_path: str, pop_init: bool, clean: bool, fitinbetween: bool, warm_start: bool
) -> bool:
    n = 500
    dx = 3
    X = np.random.random((n, dx))
    y = np.random.randn(n, 1)

    xcs = xcsf.XCS(x_dim=dx, pop_size=5, max_trials=1000, pop_init=pop_init)
    xcs.fit(X, y, verbose=False)

    # Initial, “too large” population.
    json0 = xcs.json()
    pop0 = json.loads(json0)

    # “Pruning”.
    pop1 = deepcopy(pop0)
    del pop1["classifiers"][0]
    json1 = json.dumps(pop1)
    (tmp_path / "pset1.json").write_text(json1)

    if fitinbetween:
        xcs.fit(X, y, warm_start=warm_start, verbose=False)

    xcs.json_read(str(tmp_path / "pset1.json"), clean=clean)

    # Pipe through `loads` b/c that was done above as well.
    json2 = json.dumps(json.loads(xcs.json()))

    list1 = json.loads(json1)["classifiers"]
    list2 = json.loads(json2)["classifiers"]

    if len(list1) != len(list2):
        return False
    unequal = False
    for cl1, cl2 in zip(list1, list2):
        # If there is any difference, …
        if _compare_dicts(cl1, cl2):
            unequal = True
            break
    return not unequal


@pytest.mark.parametrize(
    ("pop_init", "clean", "fitinbetween", "warm_start"),
    [
        (False, True, False, False),
        (False, True, True, False),
        (False, True, True, True),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ],
)
def test_pop_replace(
    tmp_path: str, pop_init: bool, clean: bool, fitinbetween: bool, warm_start: bool
):
    for seed in range(19):
        np.random.seed(seed)
        assert _test_pop_replace(tmp_path, pop_init, clean, fitinbetween, warm_start), (
            f"failed at seed {seed}"
        )
