#!/usr/bin/python3
#
# Copyright (C) 2019--2022 Richard Preen <rpreen@gmail.com>
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

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm

import xcsf

###############################
# Load training and test data
###############################

# Load USPS data from https://www.openml.org/d/41082
data = fetch_openml(data_id=41082)  # 256 features, 10 classes, 9298 instances
INPUT_HEIGHT: Final[int] = 16
INPUT_WIDTH: Final[int] = 16
INPUT_CHANNELS: Final[int] = 1

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.1
)

# numpy
X_train = np.asarray(X_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)
y_train = np.asarray(y_train, dtype=np.int16)
y_test = np.asarray(y_test, dtype=np.int16)

# USPS labels start at 1
y_train = np.subtract(y_train, 1)
y_test = np.subtract(y_test, 1)

# scale features [0,1]
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# one hot encode labels
onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
onehot_encoder.fit(y_train.reshape(-1, 1))
y_train = onehot_encoder.transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.transform(y_test.reshape(-1, 1))

# 10% of training for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

print(f"X_train shape = {np.shape(X_train)}")
print(f"y_train shape = {np.shape(y_train)}")
print(f"X_val shape = {np.shape(X_val)}")
print(f"y_val shape = {np.shape(y_val)}")
print(f"X_test shape = {np.shape(X_test)}")
print(f"y_test shape = {np.shape(y_test)}")

# get number of input and output variables
X_DIM: Final[int] = np.shape(X_train)[1]
Y_DIM: Final[int] = np.shape(y_train)[1]

###################
# Initialise XCSF
###################

xcs: xcsf.XCS = xcsf.XCS(X_DIM, Y_DIM, 1)  # initialise for supervised learning

xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.MAX_TRIALS = 1000  # number of trials per fit()
xcs.LOSS_FUNC = "onehot"  # one-hot encoding classification error
xcs.E0 = 0.01  # 1% target error
xcs.ALPHA = 1
xcs.BETA = 0.05
xcs.NU = 5
xcs.THETA_EA = 100
xcs.THETA_DEL = 100
xcs.action("integer")  # (dummy) integer actions

ACTIVATION: Final[str] = "selu"
SGD_WEIGHTS: Final[bool] = True
EVOLVE_WEIGHTS: Final[bool] = True
EVOLVE_CONNECT: Final[bool] = True
EVOLVE_ETA: Final[bool] = True
EVOLVE_NEURONS: Final[bool] = True
ETA: Final[float] = 0.01
ETA_MIN: Final[float] = 0.00001
MOMENTUM: Final[float] = 0.9
DECAY: Final[float] = 0
N_INIT: Final[int] = 5
N_MAX: Final[int] = 100
MAX_GROW: Final[int] = 1

CONDITION_LAYERS: Final[dict] = {
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
}
xcs.condition("neural", CONDITION_LAYERS)  # neural network conditions

LAYER_CONV: Final[dict] = {
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

LAYER_MAXPOOL: Final[dict] = {
    "type": "maxpool",
    "stride": 2,
    "size": 2,
    "pad": 0,
    "height": INPUT_HEIGHT,
    "width": INPUT_WIDTH,
    "channels": INPUT_CHANNELS,
}

LAYER_CONNECTED: Final[dict] = {
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

PREDICTION_LAYERS: Final[dict] = {
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
}
xcs.prediction("neural", PREDICTION_LAYERS)  # neural network predictions

xcs.print_params()

##################################
# Run experiment
##################################

N: Final[int] = 100  # 100,000 trials
trials: np.ndarray = np.zeros(N)
psize: np.ndarray = np.zeros(N)
msize: np.ndarray = np.zeros(N)
train_err: np.ndarray = np.zeros(N)
val_err: np.ndarray = np.zeros(N)
val_min: float = 1000  # minimum validation error observed
val_trial: int = 0  # number of trials at validation minimum

bar = tqdm(total=N)  # progress bar
for i in range(N):
    # train
    train_err[i] = xcs.fit(X_train, y_train, True)  # True = shuffle
    trials[i] = xcs.time()  # number of learning trials so far
    psize[i] = xcs.pset_size()  # current population size
    msize[i] = xcs.mset_size()  # avg match set size
    # checkpoint lowest validation error
    val_err[i] = xcs.score(X_val, y_val, 1000)  # use maximum of 1000 samples
    if val_err[i] < val_min:
        xcs.store()
        val_min = val_err[i]
        val_trial = trials[i]
    status = (  # update status
        f"trials={trials[i]:.0f} "
        f"train_err={train_err[i]:.5f} "
        f"val_err={val_err[i]:.5f} "
        f"psize={psize[i]:.1f} "
        f"msize={msize[i]:.1f}"
    )
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# final XCSF test score
print("*****************************")
print(f"Restoring system from trial {val_trial:.0f} with val_mse={val_min:.5f}")
xcs.retrieve()
pred = xcs.predict(X_test)  # soft max predictions
pred = np.argmax(pred, axis=1)  # select most likely class
pred = onehot_encoder.fit_transform(pred.reshape(-1, 1))
inv_y_test = onehot_encoder.inverse_transform(y_test)
inv_pred = onehot_encoder.inverse_transform(pred)
print(classification_report(inv_y_test, inv_pred, digits=4))

# plot XCSF learning performance
plt.figure(figsize=(10, 6))
plt.plot(trials, train_err, label="Train Error")
plt.plot(trials, val_err, label="Validation Error")
plt.grid(linestyle="dotted", linewidth=1)
plt.axhline(y=xcs.E0, xmin=0, xmax=1, linestyle="dashed", color="k")
plt.title("XCSF Training Performance", fontsize=14)
plt.xlabel("Trials", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.xlim([0, N * xcs.MAX_TRIALS])
plt.legend()
plt.show()
