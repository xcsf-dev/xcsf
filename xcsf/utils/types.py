#!/usr/bin/python3
#
# Copyright (C) 2021 Richard Preen <rpreen@gmail.com>
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

"""XCSF Python types for static checking."""

from __future__ import annotations

from typing import Dict, Literal, TypedDict, Union

EATypes = Union[Literal["roulette"], Literal["tournament"]]

ActionTypes = Union[Literal["integer"], Literal["neural"]]

ConditionTypes = Union[
    Literal["dummy"],
    Literal["hyperrectangle"],
    Literal["hyperellipsoid"],
    Literal["neural"],
    Literal["tree_gp"],
    Literal["dgp"],
    Literal["ternary"],
    Literal["rule_dgp"],
    Literal["rule_neural"],
    Literal["rule_network"],
]

PredictionTypes = Union[
    Literal["constant"],
    Literal["nlms_linear"],
    Literal["nlms_quadratic"],
    Literal["rls_linear"],
    Literal["rls_quadratic"],
    Literal["neural"],
]

LossTypes = Union[
    Literal["mae"],
    Literal["mse"],
    Literal["rmse"],
    Literal["log"],
    Literal["binary_log"],
    Literal["onehot"],
    Literal["huber"],
]


class ConditionCSRArgs(TypedDict, total=False):
    """Center-spread condition arguments."""

    eta: float
    max: float
    min: float
    spread_min: float


class ConditionGPArgs(TypedDict, total=False):
    """Tree-GP condition arguments."""

    init_depth: int
    max: float
    max_len: int
    min: float
    n_constants: int


class ConditionDGPArgs(TypedDict, total=False):
    """DGP condition arguments."""

    evolve_cycles: bool
    max_k: int
    max_t: int
    n: int


class ConditionTernaryArgs(TypedDict, total=False):
    """Ternary condition arguments."""

    bits: int
    p_dontcare: float


class PredictionNLMSArgs(TypedDict, total=False):
    """NLMS prediction arguments."""

    eta: float
    eta_min: float
    evolve_eta: bool
    x0: float


class PredictionRLSArgs(TypedDict, total=False):
    """RLS prediction arguments."""

    rls_lambda: float
    rls_scale_factor: float
    x0: float


NeuralActivationType = Union[
    Literal["logistic"],
    Literal["relu"],
    Literal["tanh"],
    Literal["linear"],
    Literal["gaussian"],
    Literal["sin"],
    Literal["cos"],
    Literal["softplus"],
    Literal["leaky"],
    Literal["selu"],
    Literal["loggy"],
    Literal["softmax"],
]

NeuralLayerType = Union[
    Literal["connected"],
    Literal["dropout"],
    Literal["noise"],
    Literal["softmax"],
    Literal["recurrent"],
    Literal["lstm"],
    Literal["maxpool"],
    Literal["convolutional"],
    Literal["avgpool"],
    Literal["upsamples"],
]


class NeuralLayerArgs(TypedDict, total=False):
    """Neural network layer arguments."""

    activation: NeuralActivationType
    channels: int
    decay: float
    eta: float
    eta_min: float
    evolve_connect: bool
    evolve_eta: bool
    evolve_functions: bool
    evolve_neurons: bool
    evolve_weights: bool
    height: int
    max_neuron_grow: int
    momentum: float
    n_init: int
    n_max: int
    pad: int
    probability: float
    recurrent_activation: str
    scale: float
    sgd_weights: bool
    size: int
    stride: int
    type: NeuralLayerType
    width: int


ActionArgs = Dict[str, NeuralLayerArgs]

ConditionArgs = Union[
    ConditionTernaryArgs,
    ConditionCSRArgs,
    ConditionGPArgs,
    ConditionDGPArgs,
    Dict[str, NeuralLayerArgs],
]

PredictionArgs = Union[
    PredictionNLMSArgs, PredictionRLSArgs, Dict[str, NeuralLayerArgs]
]
