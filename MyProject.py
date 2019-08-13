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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm

# load example data set
data = datasets.load_boston()

# split into training and test sets
train_X, test_X, train_Y, test_Y = train_test_split(data.data, data.target, test_size = 0.1, random_state = 5)

# scale [-1,1]
train_X = minmax_scale(train_X, feature_range=(-1,1))
train_Y = minmax_scale(train_Y, feature_range=(-1,1))
test_X = minmax_scale(test_X, feature_range=(-1,1))
test_Y = minmax_scale(test_Y, feature_range=(-1,1))

# XCSF inputs must be 2D numpy arrays
if(len(np.shape(train_Y)) == 1):
    train_Y = np.reshape(train_Y, (train_Y.shape[0], 1))
if(len(np.shape(test_Y)) == 1):
    test_Y = np.reshape(test_Y, (test_Y.shape[0], 1))

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
xcs.POP_SIZE = 1000
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.COND_TYPE = 3 # tree-GP conditions
xcs.PRED_TYPE = 4 # neural network predictors
xcs.HIDDEN_NEURON_ACTIVATION = 0 # logistic
xcs.NUM_HIDDEN_NEURONS = 10

##################################
# Example plotting in matplotlib
##################################

n = 50 # 50,000 evaluations
evals = np.zeros(n)
psize = np.zeros(n)
train_mse = np.zeros(n)
test_mse = np.zeros(n)
bar = tqdm(total=n) # progress bar
for i in range(n):
    # train
    xcs.fit(train_X, train_Y, True) # True = shuffle
    # get training error
    pred = xcs.predict(train_X)
    train_mse[i] = mean_squared_error(pred, train_Y)
    # get testing error
    pred = xcs.predict(test_X)
    test_mse[i] = mean_squared_error(pred, test_Y)
    evals[i] = xcs.time() # number of evaluations so far
    psize[i] = xcs.pop_num() # current population size
    # update status
    status = ("evals=%d train_mse=%.5f test_mse=%.5f popsize=%d sam0=%.3f sam1=%.3f" %
        (evals[i], train_mse[i], test_mse[i], psize[i], xcs.pop_avg_mu(0), xcs.pop_avg_mu(1)))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()
print('XCSF MSE = %.4f' % (test_mse[n-1]))

# compare with linear regression
lm = LinearRegression()
lm.fit(train_X, train_Y)
lm_pred = lm.predict(test_X)
lm_mse = mean_squared_error(lm_pred, test_Y)
print('Linear regression MSE = %.4f' % (lm_mse))

# compare with MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate='adaptive',
max_iter=1000, learning_rate_init=0.01, alpha=0.01) 
mlp.fit(train_X, train_Y.ravel())
mlp_pred = mlp.predict(test_X)
mlp_mse = mean_squared_error(mlp_pred, test_Y)
print('MLP Regressor MSE = %.4f' % (mlp_mse))

# plot XCSF learning performance
plt.figure(figsize=(10,6))
plt.plot(evals, train_mse, label='Train')
plt.plot(evals, test_mse, label='Test')
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.EPS_0, xmin=0.0, xmax=1.0, color='r')
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Evaluations', fontsize=12)
plt.xlim([0,n*xcs.MAX_TRIALS])
plt.ylabel('Mean squared error', fontsize=12)
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
