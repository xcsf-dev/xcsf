#!/usr/bin/python3
#
# Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
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
import xcsf.xcsf as xcsf
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
np.set_printoptions(suppress=True)

# Load data from https://www.openml.org/d/189
data = fetch_openml(data_id=189)

# split into training and test sets
train_X, test_X, train_Y, test_Y = train_test_split(data.data, data.target, test_size=0.1)

# reshape into 2D numpy arrays
if(len(np.shape(train_Y)) == 1):
    train_Y = np.reshape(train_Y, (train_Y.shape[0], 1))
if(len(np.shape(test_Y)) == 1):
    test_Y = np.reshape(test_Y, (test_Y.shape[0], 1))

# scale [0,1]
train_X = minmax_scale(train_X, feature_range=(0,1))
train_Y = minmax_scale(train_Y, feature_range=(0,1))
test_X = minmax_scale(test_X, feature_range=(0,1))
test_Y = minmax_scale(test_Y, feature_range=(0,1))

print("train_X shape = "+str(np.shape(train_X)))
print("train_Y shape = "+str(np.shape(train_Y)))
print("test_X shape = "+str(np.shape(test_X)))
print("test_Y shape = "+str(np.shape(test_Y)))

# get number of input and output variables
xvars = np.shape(train_X)[1]
yvars = np.shape(train_Y)[1]
print("xvars = "+str(xvars) + " yvars = " + str(yvars))

# initialise XCSF
xcs = xcsf.XCS(xvars, yvars)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.LOSS_FUNC = 1 # MSE
xcs.EPS_0 = 0.005 # target error

xcs.COND_TYPE = 2 # neural network conditions
xcs.COND_HIDDEN_NEURON_ACTIVATION = 1 # relu
xcs.COND_NUM_HIDDEN_NEURONS = 1
xcs.COND_MAX_HIDDEN_NEURONS = 20
xcs.COND_EVOLVE_WEIGHTS = True
xcs.COND_EVOLVE_NEURONS = True
xcs.COND_EVOLVE_FUNCTIONS = False

xcs.PRED_TYPE = 4 # neural network predictors
xcs.PRED_HIDDEN_NEURON_ACTIVATION = 1 # relu
xcs.PRED_NUM_HIDDEN_NEURONS = 200
xcs.PRED_MAX_HIDDEN_NEURONS = 200
xcs.PRED_EVOLVE_WEIGHTS = True
xcs.PRED_EVOLVE_NEURONS = False
xcs.PRED_EVOLVE_FUNCTIONS = False
xcs.PRED_EVOLVE_ETA = True
xcs.PRED_SGD_WEIGHTS = True

##################################
# Example plotting in matplotlib
##################################

n = 100 # 100,000 trials
trials = np.zeros(n)
psize = np.zeros(n)
msize = np.zeros(n)
train_mse = np.zeros(n)
test_mse = np.zeros(n)
bar = tqdm(total=n) # progress bar
for i in range(n):
    # train
    train_mse[i] = xcs.fit(train_X, train_Y, True) # True = shuffle
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # test
    test_mse[i] = xcs.score(test_X, test_Y)
    # update status
    status = ("trials=%d train_mse=%.5f test_mse=%.5f psize=%d msize=%.1f smut=%.4f pmut=%.4f emut=%.4f fmut=%.4f"
        % (trials[i], train_mse[i], test_mse[i], psize[i], msize[i], xcs.pop_avg_mu(0), xcs.pop_avg_mu(1), xcs.pop_avg_mu(2), xcs.pop_avg_mu(3)))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# final XCSF test score
pred = xcs.predict(test_X)
xcsf_mse = mean_squared_error(pred, test_Y)
print('XCSF MSE = %.4f' % (xcsf_mse))

# compare with linear regression
lm = LinearRegression()
lm.fit(train_X, train_Y)
lm_pred = lm.predict(test_X)
lm_mse = mean_squared_error(lm_pred, test_Y)
print('Linear regression MSE = %.4f' % (lm_mse))

# compare with MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(200,), activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)
mlp.fit(train_X, train_Y.ravel())
mlp_pred = mlp.predict(test_X)
mlp_mse = mean_squared_error(mlp_pred, test_Y)
print('MLP Regressor MSE = %.4f' % (mlp_mse))

# plot XCSF learning performance
plt.figure(figsize=(10,6))
plt.plot(trials, train_mse, label='Train MSE')
plt.plot(trials, test_mse, label='Test MSE')
#psize[:] = [x / xcs.POP_SIZE for x in psize] # scale for plotting
#msize[:] = [x / xcs.POP_SIZE for x in msize]
#plt.plot(trials, psize, label='Population macro-classifiers / P')
#plt.plot(trials, msize, label='Avg. match-set macro-classifiers / P')
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.EPS_0, xmin=0.0, xmax=1.0, linestyle='dashed', color='k')
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.xlim([0,n*xcs.MAX_TRIALS])
plt.legend()
plt.show()

# show some predictions vs. answers
pred = xcs.predict(test_X[:10])
print("*****************************")
print("first 10 predictions = ")
print(pred[:10])
print("*****************************")
print("first 10 answers = ")
print(test_Y[:10])
