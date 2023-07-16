#!/usr/bin/python3
#
# Copyright (C) 2019--2023 Richard Preen <rpreen@gmail.com>
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

"""
This example demonstrates the XCSF supervised learning mechanisms to perform
classification on the USPS handwritten digits dataset. Classifiers are composed
of neural network conditions and predictions. A softmax layer is used as
prediction output and labels are one-hot encoded. Similar to regression, a
single dummy action is performed such that [A] = [M].
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm

import xcsf

RANDOM_STATE: int = 1
np.random.seed(RANDOM_STATE)

###############################
# Load training and test data
###############################

# Load USPS data from https://www.openml.org/d/41082
# 256 features, 10 classes, 9298 instances
data = fetch_openml(data_id=41082, parser="auto")
INPUT_HEIGHT: int = 16
INPUT_WIDTH: int = 16
INPUT_CHANNELS: int = 1

# numpy
X = np.asarray(data.data, dtype=np.float64)
y = np.asarray(data.target, dtype=np.int16)

# scale features [0,1]
scaler = MinMaxScaler()
scaler.fit_transform(X)

# USPS labels start at 1
y = np.subtract(y, 1)

# one hot encode labels
onehot_encoder = OneHotEncoder(sparse_output=False, categories="auto")
onehot_encoder.fit(y.reshape(-1, 1))
y = onehot_encoder.transform(y.reshape(-1, 1))

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE
)

# 10% of training for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
)

# get number of input and output variables
X_DIM: int = np.shape(X_train)[1]
Y_DIM: int = np.shape(y_train)[1]

print(f"X_train shape = {np.shape(X_train)}")
print(f"y_train shape = {np.shape(y_train)}")
print(f"X_val shape = {np.shape(X_val)}")
print(f"y_val shape = {np.shape(y_val)}")
print(f"X_test shape = {np.shape(X_test)}")
print(f"y_test shape = {np.shape(y_test)}")

###################
# Initialise XCSF
###################

ACTIVATION: str = "selu"
SGD_WEIGHTS: bool = True
EVOLVE_WEIGHTS: bool = True
EVOLVE_CONNECT: bool = True
EVOLVE_ETA: bool = True
EVOLVE_NEURONS: bool = True
ETA: float = 0.01
ETA_MIN: float = 0.00001
MOMENTUM: float = 0.9
DECAY: float = 0
N_INIT: int = 16
N_MAX: int = 100
MAX_GROW: int = 1

LAYER_CONV: dict = {
    "type": "convolutional",
    "activation": ACTIVATION,
    "sgd_weights": SGD_WEIGHTS,
    "evolve_weights": EVOLVE_WEIGHTS,
    "evolve_connect": EVOLVE_CONNECT,
    "evolve_eta": EVOLVE_ETA,
    "evolve_neurons": EVOLVE_NEURONS,
    "max_neuron_grow": MAX_GROW,
    "eta": ETA,
    "eta_min": ETA_MIN,
    "momentum": MOMENTUM,
    "decay": DECAY,
    "n_init": N_INIT,
    "n_max": N_MAX,
    "stride": 1,
    "size": 3,
    "pad": 1,
    "height": INPUT_HEIGHT,
    "width": INPUT_WIDTH,
    "channels": INPUT_CHANNELS,
}

LAYER_MAXPOOL: dict = {
    "type": "maxpool",
    "stride": 2,
    "size": 2,
    "pad": 0,
    "height": INPUT_HEIGHT,
    "width": INPUT_WIDTH,
    "channels": INPUT_CHANNELS,
}

LAYER_CONNECTED: dict = {
    "type": "connected",
    "activation": ACTIVATION,
    "sgd_weights": SGD_WEIGHTS,
    "evolve_weights": EVOLVE_WEIGHTS,
    "evolve_connect": EVOLVE_CONNECT,
    "evolve_eta": EVOLVE_ETA,
    "evolve_neurons": EVOLVE_NEURONS,
    "max_neuron_grow": MAX_GROW,
    "eta": ETA,
    "eta_min": ETA_MIN,
    "momentum": MOMENTUM,
    "decay": DECAY,
    "n_init": N_INIT,
    "n_max": N_MAX,
}

MAX_TRIALS: int = 1000  # number of trials per fit()
E0: float = 0.01  # target error

xcs = xcsf.XCS(
    x_dim=X_DIM,
    y_dim=Y_DIM,
    n_actions=1,
    omp_num_threads=12,
    random_state=RANDOM_STATE,
    pop_size=100,
    max_trials=MAX_TRIALS,
    perf_trials=1000,
    loss_func="onehot",  # one-hot encoding classification error
    e0=E0,
    alpha=1,
    beta=0.05,
    delta=0.1,
    theta_del=100,
    nu=5,
    ea={
        "select_type": "roulette",
        "theta_ea": 100,
        "lambda": 2,
        "p_crossover": 0.8,
        "err_reduc": 1,
        "fit_reduc": 0.1,
        "subsumption": False,
        "pred_reset": False,
    },
    action={
        "type": "integer",
    },
    condition={
        "type": "neural",
        "args": {
            "layer_0": {  # hidden layer
                "type": "connected",
                "activation": ACTIVATION,
                "evolve_weights": EVOLVE_WEIGHTS,
                "evolve_connect": EVOLVE_CONNECT,
                "evolve_neurons": EVOLVE_NEURONS,
                "n_init": 1,
                "n_max": N_MAX,
                "max_neuron_grow": MAX_GROW,
            },
            "layer_1": {  # output layer
                "type": "connected",
                "activation": "linear",
                "evolve_weights": EVOLVE_WEIGHTS,
                "evolve_connect": EVOLVE_CONNECT,
                "n_init": 1,
            },
        },
    },
    prediction={
        "type": "neural",
        "args": {
            "layer_0": LAYER_CONV,
            "layer_1": LAYER_MAXPOOL,
            "layer_2": LAYER_CONV,
            "layer_3": LAYER_MAXPOOL,
            "layer_4": LAYER_CONNECTED,
            "layer_out1": {  # output layer - softmax composed of two layers
                "type": "connected",
                "activation": "linear",
                "sgd_weights": SGD_WEIGHTS,
                "evolve_weights": EVOLVE_WEIGHTS,
                "evolve_connect": EVOLVE_CONNECT,
                "evolve_eta": EVOLVE_ETA,
                "eta": ETA,
                "eta_min": ETA_MIN,
                "momentum": MOMENTUM,
                "decay": DECAY,
                "n_init": Y_DIM,
            },
            "layer_out2": {  # output layer - softmax composed of two layers
                "type": "softmax",
                "scale": 1,
            },
        },
    },
)

print(json.dumps(xcs.internal_params(), indent=4))

##################################
# Run experiment
##################################

# In this example, training is divided into multiple fit() calls in order to
# maintain control within Python.
# See the regression example for a simpler scheme with a single call to fit().

N: int = 200  # 200,000 trials
val_min: float = 1000  # minimum validation error observed
val_trial: int = 0  # number of trials at validation minimum

bar = tqdm(total=N)  # progress bar
for _ in range(N):
    xcs.fit(
        X_train,
        y_train,
        shuffle=True,  # randomly draw samples
        verbose=False,  # silence output
        warm_start=True,  # use existing population
        validation_data=(X_val, y_val),
    )
    metrics = xcs.get_metrics()
    # checkpoint lowest validation error
    if metrics["val"][-1] < val_min:
        xcs.store()
        val_min = metrics["val"][-1]
        val_trial = metrics["trials"][-1]
    status = (  # update status
        f"trials={metrics['trials'][-1]:.0f} "
        f"train_err={metrics['train'][-1]:.5f} "
        f"val_err={metrics['val'][-1]:.5f} "
        f"psize={metrics['psize'][-1]:.1f} "
        f"msize={metrics['msize'][-1]:.1f}"
    )
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

##################################
# final XCSF test score
##################################

print("*****************************")
print(f"Restoring system from trial {val_trial:.0f} with val_mse={val_min:.5f}")
xcs.retrieve()
pred = xcs.predict(X_test)  # soft max predictions
pred = np.argmax(pred, axis=1)  # select most likely class
pred = onehot_encoder.fit_transform(pred.reshape(-1, 1))
inv_y_test = onehot_encoder.inverse_transform(y_test)
inv_pred = onehot_encoder.inverse_transform(pred)
print(classification_report(inv_y_test, inv_pred, digits=4))

##################################
# plot XCSF learning performance
##################################

metrics = xcs.get_metrics()
trials = metrics["trials"]
psize = metrics["psize"]
msize = metrics["msize"]
train_err = metrics["train"]
val_err = metrics["val"]

plt.figure(figsize=(10, 6))
plt.plot(trials, train_err, label="Train Error")
plt.plot(trials, val_err, label="Validation Error")
plt.grid(linestyle="dotted", linewidth=1)
plt.axhline(y=E0, xmin=0, xmax=1, linestyle="dashed", color="k")
plt.title("XCSF Training Performance", fontsize=14)
plt.xlabel("Trials", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.xlim([0, N * MAX_TRIALS])
plt.legend()
plt.show()
