#!/usr/bin/python3
#
# Copyright (C) 2019--2021 Richard Preen <rpreen@gmail.com>
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
This example demonstrates the use of XCSF with action sets applied to the UCI
Wisconsin breast cancer classification dataset. Classifiers are composed of
hyperrectangle conditions, linear least squares predictions, and integer
actions.
"""

from __future__ import annotations

import random
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import xcsf

data = load_breast_cancer()  # 30 features, 2 classes # 569 instances
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2
)
X_train = minmax_scale(X_train, feature_range=(0, 1))
X_test = minmax_scale(X_test, feature_range=(0, 1))
X_DIM: Final[int] = 30
N_ACTIONS: Final[int] = 2
MAX_PAYOFF: Final[float] = 1
train_len: Final[int] = len(X_train)
test_len: Final[int] = len(X_test)
print(f"train len = {train_len}, test len = {test_len}")

###################
# Initialise XCSF
###################

# constructor = (x_dim, y_dim, n_actions)
xcs: xcsf.XCS = xcsf.XCS(X_DIM, 1, N_ACTIONS)

xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 1000
xcs.PERF_TRIALS = 1000
xcs.E0 = 0.01  # target error
xcs.condition("hyperrectangle")  # hyperrectangle conditions
xcs.prediction("nlms_linear")  # linear least squares predictions
xcs.action("integer")  # integer actions

xcs.print_params()

#####################
# Execute experiment
#####################

N: Final[int] = 100  # 100,000 trials
trials: np.ndarray = np.zeros(N)
psize: np.ndarray = np.zeros(N)
msize: np.ndarray = np.zeros(N)
performance: np.ndarray = np.zeros(N)
error: np.ndarray = np.zeros(N)
bar = tqdm(total=N)  # progress bar

for i in range(N):
    for _ in range(xcs.PERF_TRIALS):
        # learning trial
        sample: int = random.randint(0, train_len - 1)
        state: np.ndarray = X_train[sample]
        answer: int = y_train[sample]
        action: int = random.randrange(N_ACTIONS)  # random action
        reward: float = MAX_PAYOFF if action == answer else 0
        xcs.fit(state, action, reward)  # update action set, run EA, etc.
        # testing trial
        sample = random.randint(0, test_len - 1)
        state = X_test[sample]
        answer = y_test[sample]
        prediction_array = xcs.predict(state.reshape(1, -1))[0]
        action = int(np.argmax(prediction_array))  # best action
        reward = MAX_PAYOFF if action == answer else 0
        performance[i] += reward
        error[i] += xcs.error(reward, True, MAX_PAYOFF)
    performance[i] /= float(xcs.PERF_TRIALS)
    error[i] /= float(xcs.PERF_TRIALS)
    trials[i] = xcs.time()  # number of trials so far
    psize[i] = xcs.pset_size()  # current population size
    msize[i] = xcs.mset_size()  # avg match set size
    status = (  # update status
        f"trials={trials[i]:.0f} "
        f"performance={performance[i]:.5f} "
        f"error={error[i]:.5f} "
        f"psize={psize[i]:.1f} "
        f"msize={msize[i]:.1f}"
    )
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

##################################
# plot XCSF learning performance
##################################

plt.figure(figsize=(10, 6))
plt.plot(trials, performance, label="Performance")
plt.plot(trials, error, label="System error")
plt.grid(linestyle="dotted", linewidth=1)
plt.title("Wisconsin Breast Cancer", fontsize=14)
plt.xlabel("Trials", fontsize=12)
plt.xlim([0, N * xcs.PERF_TRIALS])
plt.legend()
plt.show()

###################
# Confusion Matrix
###################

prediction_arrays = xcs.predict(X_test)  # generate prediction arrays
y_pred = np.argmax(prediction_arrays, axis=1)  # class with largest prediction

confusion_matrix_val = confusion_matrix(y_test, y_pred)
positions = [0, 1]
labels = ["Benign", "Malignant"]
fig = plt.figure()
subplt = fig.add_subplot(111)
csubplt = subplt.matshow(confusion_matrix_val)
plt.title("Wisconsin Breast Cancer:")
fig.colorbar(csubplt)
subplt.xaxis.set_major_locator(ticker.FixedLocator(positions))
subplt.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
subplt.yaxis.set_major_locator(ticker.FixedLocator(positions))
subplt.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

############################
# Compare with alternatives
############################

print(f"XCSF accuracy: {accuracy_score(y_test, y_pred)}")
print(f"XCSF f1: {f1_score(y_test, y_pred, average=None)}")

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    learning_rate="adaptive",
    max_iter=1000,
)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(f"MLP accuracy: {accuracy_score(y_test, y_pred)}")
print(f"MLP f1: {f1_score(y_test, y_pred, average=None)}")

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(f"Decision tree accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Decision tree f1: {f1_score(y_test, y_pred, average=None)}")
