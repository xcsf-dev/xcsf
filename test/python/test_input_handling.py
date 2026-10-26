#
# Copyright (C) 2026 Richard Preen <rpreen@gmail.com>
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

"""XCSF Python input handling tests."""

import numpy as np
import pytest

import xcsf


def make_xcs(**kwargs):
    """Construct a small, valid XCS, overriding any parameters given."""
    params = {
        "x_dim": 2,
        "y_dim": 1,
        "n_actions": 1,
        "pop_size": 50,
        "max_trials": 100,
        "perf_trials": 50,
    }
    params.update(kwargs)
    return xcsf.XCS(**params)


@pytest.fixture
def xcs():
    return make_xcs()


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    X = rng.random((20, 2))
    y = rng.random((20, 1))
    return X, y


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_valid_construction():
    make_xcs()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"x_dim": 0}, "x_dim"),
        ({"x_dim": -1}, "x_dim"),
        ({"y_dim": 0}, "y_dim"),
        ({"y_dim": -3}, "y_dim"),
        ({"n_actions": 0}, "n_actions"),
        ({"n_actions": -1}, "n_actions"),
        ({"pop_size": 0}, "pop_size"),
        ({"max_trials": 0}, "max_trials"),
        ({"perf_trials": 0}, "perf_trials"),
        ({"beta": -0.1}, "beta"),
        ({"beta": 1.5}, "beta"),
        ({"e0": -1.0}, "e0"),
        ({"x_dim": "two"}, "x_dim"),
        ({"y_dim": None}, "y_dim"),
        ({"beta": "high"}, "beta"),
    ],
)
def test_constructor_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_xcs(**kwargs)


def test_constructor_rejects_unknown_keyword():
    with pytest.raises(ValueError, match="unable to import parameter"):
        make_xcs(not_a_real_parameter=1)


## ---------------------------------------------------------------------------
## set_params
## ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"beta": -0.5}, "beta"),
        ({"beta": 2}, "beta"),
        ({"pop_size": 0}, "pop_size"),
        ({"e0": -0.01}, "e0"),
        ({"gamma": 1.5}, "gamma"),
    ],
)
def test_set_params_rejects_invalid_values(xcs, kwargs, match):
    with pytest.raises(ValueError, match=match):
        xcs.set_params(**kwargs)


def test_set_params_accepts_valid_values(xcs):
    xcs.set_params(beta=0.2, pop_size=100)
    params = xcs.get_params()
    assert params["beta"] == 0.2
    assert params["pop_size"] == 100


def test_failed_set_params_leaves_value_unchanged(xcs):
    before = xcs.get_params()["beta"]
    with pytest.raises(ValueError, match="beta"):
        xcs.set_params(beta=-1)
    assert xcs.get_params()["beta"] == before


def test_error_is_not_sticky_after_failure(xcs):
    """A failed call must not make later valid calls fail."""
    with pytest.raises(ValueError, match="beta"):
        xcs.set_params(beta=-1)
    xcs.set_params(beta=0.3)
    assert xcs.get_params()["beta"] == 0.3


def test_same_error_raised_every_time(xcs):
    for _ in range(3):
        with pytest.raises(ValueError, match="beta"):
            xcs.set_params(beta=-1)


# ---------------------------------------------------------------------------
# fit / score / predict input checking
# ---------------------------------------------------------------------------


def test_fit_valid_input_runs(xcs, data):
    X, y = data
    xcs.fit(X, y, verbose=False)


def test_fit_wrong_x_dim(xcs, data):
    _, y = data
    x_bad = np.zeros((20, 3))  # x_dim is 2
    with pytest.raises(ValueError, match="(?i)x_dim|shape|dim"):
        xcs.fit(x_bad, y, verbose=False)


def test_fit_wrong_y_dim(xcs, data):
    X, _ = data
    y_bad = np.zeros((20, 2))  # y_dim is 1
    with pytest.raises(ValueError, match="(?i)y_dim|shape|dim"):
        xcs.fit(X, y_bad, verbose=False)


def test_fit_sample_count_mismatch(xcs, data):
    X, y = data
    with pytest.raises(ValueError, match="(?i)sample|shape|size|length"):
        xcs.fit(X, y[:10], verbose=False)


def test_fit_bad_validation_data_shape(xcs, data):
    X, y = data
    with pytest.raises(ValueError, match="(?i)x_dim|shape|dim"):
        xcs.fit(
            X, y, verbose=False, validation_data=(np.zeros((5, 3)), np.zeros((5, 1)))
        )


def test_score_wrong_x_dim(xcs, data):
    _, y = data
    with pytest.raises(ValueError, match="(?i)x_dim|shape|dim"):
        xcs.score(np.zeros((20, 5)), y)


def test_score_sample_count_mismatch(xcs, data):
    X, y = data
    with pytest.raises(ValueError, match="(?i)sample|shape|size|length"):
        xcs.score(X, y[:5])


def test_predict_wrong_x_dim(xcs):
    with pytest.raises(ValueError, match="(?i)x_dim|shape|dim"):
        xcs.predict(np.zeros((4, 7)))


def test_predict_cover_wrong_length(xcs, data):
    X, _ = data
    with pytest.raises(ValueError, match="(?i)cover|y_dim|length|shape"):
        xcs.predict(X, cover=np.zeros(5))  # y_dim is 1


def test_object_usable_after_failed_fit(xcs, data):
    X, y = data
    with pytest.raises(ValueError, match="(?i)x_dim|shape|dim"):
        xcs.fit(np.zeros((20, 3)), y, verbose=False)
    xcs.fit(X, y, verbose=False)  # must still work
    assert xcs.predict(X).shape == (20, 1)


# ---------------------------------------------------------------------------
# Insert / load errors
# ---------------------------------------------------------------------------


def test_json_insert_invalid_json(xcs):
    with pytest.raises(ValueError, match="Error reading JSON"):
        xcs.json_insert("{ this is not valid json")


def test_json_read_missing_file(xcs, tmp_path):
    with pytest.raises(ValueError, match="Error reading JSON"):
        xcs.json_read(str(tmp_path / "missing.json"))


def test_load_missing_file(xcs, tmp_path):
    with pytest.raises(ValueError, match="Error loading file"):
        xcs.load(str(tmp_path / "missing.bin"))
