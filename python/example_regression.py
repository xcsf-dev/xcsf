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
This example demonstrates the XCSF supervised learning mechanisms to perform
regression on the Boston house price dataset. Classifiers are composed of tree
GP conditions and neural network predictions. A single dummy action is
performed such that [A] = [M].
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import xcsf
np.set_printoptions(suppress=True)

###############################
# Load training and test data
###############################

# Load data from https://www.openml.org/d/189
data = fetch_openml(data_id=189)

# split into training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(data.data, data.target, test_size=0.1)

# numpy
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# reshape into 2D arrays
if len(np.shape(y_train)) == 1:
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

# normalise inputs (zero mean and unit variance)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# scale outputs [0,1]
y_train = minmax_scale(y_train, feature_range=(0, 1))
y_test = minmax_scale(y_test, feature_range=(0, 1))

# get number of input and output variables
X_DIM = np.shape(X_train)[1]
Y_DIM = np.shape(y_train)[1]

# 10% of training for validation
X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.1)

print('X_train shape = ' + str(np.shape(X_train)))
print('y_train shape = ' + str(np.shape(y_train)))
print('X_val shape = ' + str(np.shape(X_val)))
print('y_val shape = ' + str(np.shape(y_val)))
print('X_test shape = ' + str(np.shape(X_test)))
print('y_test shape = ' + str(np.shape(y_test)))

###################
# Initialise XCSF
###################

xcs = xcsf.XCS(X_DIM, Y_DIM, 1) # initialise XCSF for supervised learning

xcs.OMP_NUM_THREADS = 8 # number of CPU cores to use
xcs.POP_SIZE = 500 # maximum population size
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.LOSS_FUNC = 'mse' # mean squared error
xcs.E0 = 0.005 # target error
xcs.ALPHA = 1 # accuracy offset
xcs.NU = 20 # accuracy slope
xcs.THETA_EA = 50 # EA invocation frequency
xcs.THETA_DEL = 50 # min experience before fitness used in deletion
xcs.BETA = 0.1 # update rate for error, etc.
xcs.action('integer') # (dummy) integer actions

tree_args = {
    'min': 0, # minimum value of a constant
    'max': 1, # maximum value of a constant
    'n-constants': 100, # number of (global) constants
    'init-depth': 5, # initial tree depth
    'max-len': 10000, # maximum initial length
}
xcs.condition('tree-gp', tree_args) # GP tree conditions

prediction_layers = {
    'layer_0': { # hidden layer
        'type': 'connected',
        'activation': 'relu',
        'sgd-weights': True,
        'evolve-weights': True,
        'evolve-connect': True,
        'evolve-eta': True,
        'eta': 0.1,
        'eta-min': 0.000001,
        'momentum': 0.9,
        'n-init': 10,
        'n-max': 10,
    },
    'layer_1': { # output layer
        'type': 'connected',
        'activation': 'softplus',
        'sgd-weights': True,
        'evolve-weights': True,
        'evolve-connect': True,
        'evolve-eta': True,
        'eta': 0.1,
        'eta-min': 0.000001,
        'momentum': 0.9,
        'n-init': Y_DIM,
    }
}
xcs.prediction('neural', prediction_layers) # neural network predictions

xcs.print_params()

#################
# Run experiment
#################

N = 200 # 200,000 trials
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
train_mse = np.zeros(N)
val_mse = np.zeros(N)

VAL_PERIOD = 10 # validation mean moving average length
val_min = 999999 # minimum validation error
val_trial = 0 # trial number the system was checkpointed

bar = tqdm(total=N) # progress bar
for i in range(N):
    # train
    train_mse[i] = xcs.fit(X_train, y_train, True) # True = shuffle
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pset_size() # current population size
    msize[i] = xcs.mset_size() # avg match set size
    # validate
    val_mse[i] = xcs.score(X_val, y_val)
    if i > VAL_PERIOD: # simple moving average of mean validation error
        val_mean = np.mean(val_mse[i - VAL_PERIOD : i])
        if val_mean < val_min: # checkpoint lowest validation error
            xcs.store()
            val_min = val_mean
            val_trial = trials[i]
    # update status
    status = ('trials=%d train_mse=%.5f val_mse=%.5f psize=%d msize=%.1f' %
              (trials[i], train_mse[i], val_mse[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# plot XCSF learning performance
plt.figure(figsize=(10, 6))
plt.plot(trials, train_mse, label='Train MSE')
plt.plot(trials, val_mse, label='Validation MSE')
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.E0, xmin=0, xmax=1, linestyle='dashed', color='k')
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.xlim([0, N * xcs.MAX_TRIALS])
plt.legend()
plt.show()

############################
# Compare with alternatives
############################

# final XCSF test score
print('*****************************')
print('Restoring system from trial %d with val_mse=%.5f' % (val_trial, val_min))
xcs.retrieve()
xcsf_pred = xcs.predict(X_test)
xcsf_mse = mean_squared_error(xcsf_pred, y_test)
print('XCSF Test MSE = %.4f' % (xcsf_mse))

# compare with linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm_pred = lm.predict(X_test)
lm_mse = mean_squared_error(lm_pred, y_test)
print('Linear regression Test MSE = %.4f' % (lm_mse))

# compare with MLP regressor
X_train = np.vstack((X_train, X_val))
y_train = np.vstack((y_train, y_val))
mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                   learning_rate='adaptive', learning_rate_init=0.01,
                   max_iter=1000, alpha=0.01, validation_fraction=0.1)
mlp.fit(X_train, y_train.ravel())
mlp_pred = mlp.predict(X_test)
mlp_mse = mean_squared_error(mlp_pred, y_test)
print('MLP Regressor Test MSE = %.4f' % (mlp_mse))

#####################################
# Show some predictions vs. answers
#####################################

pred = xcs.predict(X_test[:10])
print('*****************************')
print('first 10 predictions = ')
print(pred[:10])
print('*****************************')
print('first 10 answers = ')
print(y_test[:10])
