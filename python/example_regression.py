#!/usr/bin/python3
#
# Copyright (C) 2019--2020 Richard Preen <rpreen@gmail.com>
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

"""This example demonstrates the XCSF supervised learning mechanisms to perform
regression on the Boston house price dataset. Classifiers are composed of
neural network conditions and predictions. A single dummy action is performed
such that [A] = [M]."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import xcsf.xcsf as xcsf
np.set_printoptions(suppress=True)

###############################
# Load training and test data
###############################

# Load data from https://www.openml.org/d/189
data = fetch_openml(data_id=189)

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)

# reshape into 2D numpy arrays
if len(np.shape(y_train)) == 1:
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
if len(np.shape(y_test)) == 1:
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

# normalise inputs (zero mean and unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# scale outputs [0,1]
y_train = minmax_scale(y_train, feature_range=(0, 1))
y_test = minmax_scale(y_test, feature_range=(0, 1))

print("X_train shape = "+str(np.shape(X_train)))
print("y_train shape = "+str(np.shape(y_train)))
print("X_test shape = "+str(np.shape(X_test)))
print("y_test shape = "+str(np.shape(y_test)))

# get number of input and output variables
X_DIM = np.shape(X_train)[1]
Y_DIM = np.shape(y_train)[1]
print("x_dim = "+str(X_DIM) + ", y_dim = " + str(Y_DIM))

###################
# Initialise XCSF
###################

# initialise XCSF for supervised learning
xcs = xcsf.XCS(X_DIM, Y_DIM, 1)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.LOSS_FUNC = 1 # MSE
xcs.EPS_0 = 0.004 # target error
xcs.ALPHA = 1 # accuracy offset
xcs.NU = 20 # accuracy slope
xcs.THETA_EA = 50 # EA invocation frequency
xcs.THETA_DEL = 50 # min experience before fitness used in deletion
xcs.BETA = 0.1 # update rate for error, etc.

xcs.MAX_NEURON_GROW = 1 # max neurons to add/remove per mutation event
xcs.COND_TYPE = 3 # neural network conditions
xcs.COND_OUTPUT_ACTIVATION = 3 # linear
xcs.COND_HIDDEN_ACTIVATION = 1 # relu
xcs.COND_NUM_NEURONS = [1]
xcs.COND_MAX_NEURONS = [20]
xcs.COND_EVOLVE_WEIGHTS = True
xcs.COND_EVOLVE_NEURONS = True
xcs.COND_EVOLVE_FUNCTIONS = False
xcs.COND_EVOLVE_CONNECTIVITY = True

xcs.PRED_TYPE = 5 # neural network predictors
xcs.PRED_OUTPUT_ACTIVATION = 7 # softplus
xcs.PRED_HIDDEN_ACTIVATION = 1 # relu
xcs.PRED_NUM_NEURONS = [50]
xcs.PRED_MAX_NEURONS = [50]
xcs.PRED_EVOLVE_WEIGHTS = True
xcs.PRED_EVOLVE_NEURONS = False
xcs.PRED_EVOLVE_FUNCTIONS = False
xcs.PRED_EVOLVE_CONNECTIVITY = True
xcs.PRED_EVOLVE_ETA = True
xcs.PRED_SGD_WEIGHTS = True
xcs.PRED_ETA = 0.01 # maximum gradient descent rate

xcs.print_params()

##################################
# Example plotting in matplotlib
##################################

N = 100 # 100,000 trials
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
train_mse = np.zeros(N)
test_mse = np.zeros(N)
bar = tqdm(total=N) # progress bar
for i in range(N):
    # train
    train_mse[i] = xcs.fit(X_train, y_train, True) # True = shuffle
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # test
    test_mse[i] = xcs.score(X_test, y_test)
    # update status
    status = ("trials=%d train_mse=%.5f test_mse=%.5f psize=%d msize=%.1f" %
              (trials[i], train_mse[i], test_mse[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# plot XCSF learning performance
plt.figure(figsize=(10, 6))
plt.plot(trials, train_mse, label='Train MSE')
plt.plot(trials, test_mse, label='Test MSE')
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.EPS_0, xmin=0, xmax=1, linestyle='dashed', color='k')
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
xcsf_pred = xcs.predict(X_test)
xcsf_mse = mean_squared_error(xcsf_pred, y_test)
print('XCSF MSE = %.4f' % (xcsf_mse))

# compare with linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm_pred = lm.predict(X_test)
lm_mse = mean_squared_error(lm_pred, y_test)
print('Linear regression MSE = %.4f' % (lm_mse))

# compare with MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam',
                   learning_rate='adaptive', learning_rate_init=0.01,
                   max_iter=1000, alpha=0.01)
mlp.fit(X_train, y_train.ravel())
mlp_pred = mlp.predict(X_test)
mlp_mse = mean_squared_error(mlp_pred, y_test)
print('MLP Regressor MSE = %.4f' % (mlp_mse))

#####################################
# Show some predictions vs. answers
#####################################

pred = xcs.predict(X_test[:10])
print("*****************************")
print("first 10 predictions = ")
print(pred[:10])
print("*****************************")
print("first 10 answers = ")
print(y_test[:10])
