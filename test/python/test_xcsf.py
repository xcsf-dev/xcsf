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

from collections import namedtuple

import pickle
import numpy as np
import pytest
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xcsf

# pylint: disable=redefined-outer-name,invalid-name

PATH: str = "RES_PYTEST"
SEED: int = 101

Data = namedtuple(
    "Data",
    ["x_dim", "y_dim", "x_train", "y_train", "x_val", "y_val", "x_test", "y_test"],
)


@pytest.fixture
def data() -> Data:
    """Load test data."""
    df = fetch_openml(data_id=189, as_frame=True, parser="auto")
    x = np.asarray(df.data, dtype=np.float64)
    y = np.asarray(df.target, dtype=np.float64)

    feature_scaler = StandardScaler()
    x = feature_scaler.fit_transform(x)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    output_scaler = StandardScaler()
    y = output_scaler.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=SEED
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=SEED
    )

    return Data(
        np.shape(x_train)[1],
        np.shape(y_train)[1],
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


def predictions():
    """Return list of prediction args."""
    return [
        {"type": "constant"},
        {"type": "nlms_linear"},
        {"type": "nlms_quadratic"},
        {"type": "rls_linear"},
        {"type": "rls_quadratic"},
        {"type": "neural"},
    ]


def conditions():
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
    xcs = xcsf.XCS(
        x_dim=data.x_dim,
        y_dim=data.y_dim,
        pop_size=200,
        max_trials=1000,
        random_state=SEED,
        prediction=prediction,
    )
    xcs.fit(data.x_train, data.y_train)
    a: np.ndarray = xcs.predict(data.x_test)
    b: np.ndarray = xcs.predict(data.x_test)
    assert np.all(a == b)


@pytest.mark.parametrize("condition", conditions())
@pytest.mark.parametrize("prediction", predictions())
def test_saving(data, condition, prediction):
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
        pop_size=20,
        max_trials=100,
        random_state=SEED,
        condition=condition,
        prediction=prediction,
    )

    # fit model
    xcs1.fit(data.x_train, data.y_train)

    # save with pickle
    with open("blah.pkl", "wb") as fp:
        pickle.dump(xcs1, fp)

    # load from pickle
    with open("blah.pkl", "rb") as fp:
        xcs2 = pickle.load(fp)

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
