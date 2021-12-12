from __future__ import annotations
import typing
from typing import Any, Literal, Union, TypedDict, Dict
import numpy as np

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
    eta: float
    max: float
    min: float
    spread_min: float

class ConditionGPArgs(TypedDict, total=False):
    init_depth: int
    max: float
    max_len: int
    min: float
    n_constants: int

class ConditionDGPArgs(TypedDict, total=False):
    evolve_cycles: bool
    max_k: int
    max_t: int
    n: int

class ConditionTernaryArgs(TypedDict, total=False):
    bits: int
    p_dontcare: float

class PredictionNLMSArgs(TypedDict, total=False):
    eta: float
    eta_min: float
    evolve_eta: bool
    x0: float

class PredictionRLSArgs(TypedDict, total=False):
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

class XCS:
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def action(self, arg0: ActionTypes) -> None: ...
    @typing.overload
    def action(self, arg0: ActionTypes, arg1: ActionArgs) -> None: ...
    def ae_to_classifier(self, arg0: int, arg1: int) -> None: ...
    def aset_size(self) -> float: ...
    @typing.overload
    def condition(self, arg0: ConditionTypes) -> None: ...
    @typing.overload
    def condition(self, arg0: ConditionTypes, arg1: ConditionArgs) -> None: ...
    def decision(
        self,
        arg0: np.ndarray[Any, np.dtype[np.float64]],
        arg1: bool,
    ) -> int: ...
    def end_step(self) -> None: ...
    def end_trial(self) -> None: ...
    @typing.overload
    def error(self) -> float: ...
    @typing.overload
    def error(self, arg0: float, arg1: bool, arg2: float) -> float: ...
    @typing.overload
    def fit(
        self,
        arg0: np.ndarray[Any, np.dtype[np.float64]],
        arg1: int,
        arg2: float,
    ) -> float: ...
    @typing.overload
    def fit(
        self,
        arg0: np.ndarray[Any, np.dtype[np.float64]],
        arg1: np.ndarray[Any, np.dtype[np.float64]],
        arg2: bool,
    ) -> float: ...
    @typing.overload
    def fit(
        self,
        arg0: np.ndarray[Any, np.dtype[np.float64]],
        arg1: np.ndarray[Any, np.dtype[np.float64]],
        arg2: np.ndarray[Any, np.dtype[np.float64]],
        arg3: np.ndarray[Any, np.dtype[np.float64]],
        arg4: bool,
    ) -> float: ...
    def init_step(self) -> None: ...
    def init_trial(self) -> None: ...
    def json(self, arg0: bool, arg1: bool, arg2: bool) -> str: ...
    def json_parameters(self) -> str: ...
    def load(self, arg0: str) -> int: ...
    def mfrac(self) -> float: ...
    def mset_size(self) -> float: ...
    def n_actions(self) -> int: ...
    def pred_expand(self) -> None: ...
    def predict(
        self, arg0: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...
    @typing.overload
    def prediction(self, arg0: PredictionTypes) -> None: ...
    @typing.overload
    def prediction(self, arg0: PredictionTypes, arg1: PredictionArgs) -> None: ...
    def print_params(self) -> None: ...
    def print_pset(self, arg0: bool, arg1: bool, arg2: bool) -> None: ...
    def pset_mean_cond_connections(self, arg0: int) -> float: ...
    def pset_mean_cond_layers(self) -> float: ...
    def pset_mean_cond_neurons(self, arg0: int) -> float: ...
    def pset_mean_cond_size(self) -> float: ...
    def pset_mean_pred_connections(self, arg0: int) -> float: ...
    def pset_mean_pred_eta(self, arg0: int) -> float: ...
    def pset_mean_pred_layers(self) -> float: ...
    def pset_mean_pred_neurons(self, arg0: int) -> float: ...
    def pset_mean_pred_size(self) -> float: ...
    def pset_num(self) -> int: ...
    def pset_size(self) -> int: ...
    def retrieve(self) -> None: ...
    def save(self, arg0: str) -> int: ...
    @typing.overload
    def score(
        self,
        arg0: np.ndarray[Any, np.dtype[np.float64]],
        arg1: np.ndarray[Any, np.dtype[np.float64]],
    ) -> float: ...
    @typing.overload
    def score(
        self,
        arg0: np.ndarray[Any, np.dtype[np.float64]],
        arg1: np.ndarray[Any, np.dtype[np.float64]],
        arg2: int,
    ) -> float: ...
    def seed(self, arg0: int) -> None: ...
    def store(self) -> None: ...
    def time(self) -> int: ...
    def update(self, arg0: float, arg1: bool) -> None: ...
    def version_build(self) -> int: ...
    def version_major(self) -> int: ...
    def version_minor(self) -> int: ...
    def x_dim(self) -> int: ...
    def y_dim(self) -> int: ...
    @property
    def ALPHA(self) -> float:
        """
        :type: float
        """
    @ALPHA.setter
    def ALPHA(self, arg1: float) -> None:
        pass
    @property
    def BETA(self) -> float:
        """
        :type: float
        """
    @BETA.setter
    def BETA(self, arg1: float) -> None:
        pass
    @property
    def COMPACTION(self) -> bool:
        """
        :type: bool
        """
    @COMPACTION.setter
    def COMPACTION(self, arg1: bool) -> None:
        pass
    @property
    def DELTA(self) -> float:
        """
        :type: float
        """
    @DELTA.setter
    def DELTA(self, arg1: float) -> None:
        pass
    @property
    def E0(self) -> float:
        """
        :type: float
        """
    @E0.setter
    def E0(self, arg1: float) -> None:
        pass
    @property
    def EA_PRED_RESET(self) -> bool:
        """
        :type: bool
        """
    @EA_PRED_RESET.setter
    def EA_PRED_RESET(self, arg1: bool) -> None:
        pass
    @property
    def EA_SELECT_SIZE(self) -> float:
        """
        :type: float
        """
    @EA_SELECT_SIZE.setter
    def EA_SELECT_SIZE(self, arg1: float) -> None:
        pass
    @property
    def EA_SELECT_TYPE(self) -> EATypes:
        """
        :type: Union[Literal["roulette"], Literal["tournament"]]
        """
    @EA_SELECT_TYPE.setter
    def EA_SELECT_TYPE(self, arg1: EATypes) -> None:
        pass
    @property
    def EA_SUBSUMPTION(self) -> bool:
        """
        :type: bool
        """
    @EA_SUBSUMPTION.setter
    def EA_SUBSUMPTION(self, arg1: bool) -> None:
        pass
    @property
    def ERR_REDUC(self) -> float:
        """
        :type: float
        """
    @ERR_REDUC.setter
    def ERR_REDUC(self, arg1: float) -> None:
        pass
    @property
    def FIT_REDUC(self) -> float:
        """
        :type: float
        """
    @FIT_REDUC.setter
    def FIT_REDUC(self, arg1: float) -> None:
        pass
    @property
    def GAMMA(self) -> float:
        """
        :type: float
        """
    @GAMMA.setter
    def GAMMA(self, arg1: float) -> None:
        pass
    @property
    def HUBER_DELTA(self) -> float:
        """
        :type: float
        """
    @HUBER_DELTA.setter
    def HUBER_DELTA(self, arg1: float) -> None:
        pass
    @property
    def INIT_ERROR(self) -> float:
        """
        :type: float
        """
    @INIT_ERROR.setter
    def INIT_ERROR(self, arg1: float) -> None:
        pass
    @property
    def INIT_FITNESS(self) -> float:
        """
        :type: float
        """
    @INIT_FITNESS.setter
    def INIT_FITNESS(self, arg1: float) -> None:
        pass
    @property
    def LAMBDA(self) -> int:
        """
        :type: int
        """
    @LAMBDA.setter
    def LAMBDA(self, arg1: int) -> None:
        pass
    @property
    def LOSS_FUNC(self) -> LossTypes:
        """
        :type: Union[Literal["mae"], Literal["mse"], Literal["rmse"], Literal["log"],
                     Literal["binary_log"], Literal["onehot"], Literal["huber"]]
        """
    @LOSS_FUNC.setter
    def LOSS_FUNC(self, arg1: LossTypes) -> None:
        pass
    @property
    def MAX_TRIALS(self) -> int:
        """
        :type: int
        """
    @MAX_TRIALS.setter
    def MAX_TRIALS(self, arg1: int) -> None:
        pass
    @property
    def M_PROBATION(self) -> int:
        """
        :type: int
        """
    @M_PROBATION.setter
    def M_PROBATION(self, arg1: int) -> None:
        pass
    @property
    def NU(self) -> float:
        """
        :type: float
        """
    @NU.setter
    def NU(self, arg1: float) -> None:
        pass
    @property
    def OMP_NUM_THREADS(self) -> int:
        """
        :type: int
        """
    @OMP_NUM_THREADS.setter
    def OMP_NUM_THREADS(self, arg1: int) -> None:
        pass
    @property
    def PERF_TRIALS(self) -> int:
        """
        :type: int
        """
    @PERF_TRIALS.setter
    def PERF_TRIALS(self, arg1: int) -> None:
        pass
    @property
    def POP_INIT(self) -> bool:
        """
        :type: bool
        """
    @POP_INIT.setter
    def POP_INIT(self, arg1: bool) -> None:
        pass
    @property
    def POP_SIZE(self) -> int:
        """
        :type: int
        """
    @POP_SIZE.setter
    def POP_SIZE(self, arg1: int) -> None:
        pass
    @property
    def P_CROSSOVER(self) -> float:
        """
        :type: float
        """
    @P_CROSSOVER.setter
    def P_CROSSOVER(self, arg1: float) -> None:
        pass
    @property
    def P_EXPLORE(self) -> float:
        """
        :type: float
        """
    @P_EXPLORE.setter
    def P_EXPLORE(self, arg1: float) -> None:
        pass
    @property
    def SET_SUBSUMPTION(self) -> bool:
        """
        :type: bool
        """
    @SET_SUBSUMPTION.setter
    def SET_SUBSUMPTION(self, arg1: bool) -> None:
        pass
    @property
    def STATEFUL(self) -> bool:
        """
        :type: bool
        """
    @STATEFUL.setter
    def STATEFUL(self, arg1: bool) -> None:
        pass
    @property
    def TELETRANSPORTATION(self) -> int:
        """
        :type: int
        """
    @TELETRANSPORTATION.setter
    def TELETRANSPORTATION(self, arg1: int) -> None:
        pass
    @property
    def THETA_DEL(self) -> int:
        """
        :type: int
        """
    @THETA_DEL.setter
    def THETA_DEL(self, arg1: int) -> None:
        pass
    @property
    def THETA_EA(self) -> float:
        """
        :type: float
        """
    @THETA_EA.setter
    def THETA_EA(self, arg1: float) -> None:
        pass
    @property
    def THETA_SUB(self) -> int:
        """
        :type: int
        """
    @THETA_SUB.setter
    def THETA_SUB(self, arg1: int) -> None:
        pass
    pass
