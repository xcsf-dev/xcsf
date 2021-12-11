from __future__ import annotations
import typing
from typing import Any
import numpy

_Shape = typing.Tuple[int, ...]

class XCS:
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @typing.overload
    def action(self, arg0: str) -> None: ...
    @typing.overload
    def action(self, arg0: str, arg1: dict) -> None: ...
    def ae_to_classifier(self, arg0: int, arg1: int) -> None: ...
    def aset_size(self) -> float: ...
    @typing.overload
    def condition(self, arg0: str) -> None: ...
    @typing.overload
    def condition(self, arg0: str, arg1: dict) -> None: ...
    def decision(self, arg0: numpy.ndarray[numpy.float64, Any], arg1: bool) -> int: ...
    def end_step(self) -> None: ...
    def end_trial(self) -> None: ...
    @typing.overload
    def error(self) -> float: ...
    @typing.overload
    def error(self, arg0: float, arg1: bool, arg2: float) -> float: ...
    @typing.overload
    def fit(
        self, arg0: numpy.ndarray[numpy.float64, Any], arg1: int, arg2: float
    ) -> float: ...
    @typing.overload
    def fit(
        self,
        arg0: numpy.ndarray[numpy.float64, Any],
        arg1: numpy.ndarray[numpy.float64, Any],
        arg2: bool,
    ) -> float: ...
    @typing.overload
    def fit(
        self,
        arg0: numpy.ndarray[numpy.float64, Any],
        arg1: numpy.ndarray[numpy.float64, Any],
        arg2: numpy.ndarray[numpy.float64, Any],
        arg3: numpy.ndarray[numpy.float64, Any],
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
        self, arg0: numpy.ndarray[numpy.float64, Any]
    ) -> numpy.ndarray[numpy.float64, Any]: ...
    @typing.overload
    def prediction(self, arg0: str) -> None: ...
    @typing.overload
    def prediction(self, arg0: str, arg1: dict) -> None: ...
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
        arg0: numpy.ndarray[numpy.float64, Any],
        arg1: numpy.ndarray[numpy.float64, Any],
    ) -> float: ...
    @typing.overload
    def score(
        self,
        arg0: numpy.ndarray[numpy.float64, Any],
        arg1: numpy.ndarray[numpy.float64, Any],
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
    def EA_SELECT_TYPE(self) -> str:
        """
        :type: str
        """
    @EA_SELECT_TYPE.setter
    def EA_SELECT_TYPE(self, arg1: str) -> None:
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
    def LOSS_FUNC(self) -> str:
        """
        :type: str
        """
    @LOSS_FUNC.setter
    def LOSS_FUNC(self, arg1: str) -> None:
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
