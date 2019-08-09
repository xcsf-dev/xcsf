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
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

# load example data set
data = datasets.load_boston() # regression
train_X, train_Y = data.data, data.target

# scale [-1,1]
train_X = minmax_scale(train_X, feature_range=(-1,1))
train_Y = minmax_scale(train_Y, feature_range=(-1,1))

# XCSF inputs must be 2D numpy arrays
if(len(np.shape(train_Y)) == 1):
    train_Y = np.reshape(train_Y, (train_Y.shape[0], 1))

print("train_X shape = "+str(np.shape(train_X)))
print("train_Y shape = "+str(np.shape(train_Y)))

# get number of input and output variables
xvars = np.shape(train_X)[1]
yvars = np.shape(train_Y)[1] 
print("xvars = "+str(xvars) + " yvars = " + str(yvars))

# initialise XCSF
xcs = xcsf.XCS(xvars, yvars)

# override cons.txt
xcs.POP_SIZE = 5000
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.COND_TYPE = 0 # hyperrectangle conditions
xcs.PRED_TYPE = 4 # neural network predictors
xcs.HIDDEN_NEURON_ACTIVATION = 0 # logistic

##################################
# Example plotting in matplotlib
##################################

n = 100 # 100,000 trials
evals = np.zeros(n)
psize = np.zeros(n)
mse = np.zeros(n)
for i in range(n):
    xcs.fit(train_X, train_Y, True)
    pred = xcs.predict(train_X)
    mse[i] = (np.square(pred - train_Y)).mean(axis=0)
    evals[i] = xcs.time()
    psize[i] = xcs.pop_num()
    print("%d %.5f %d" % (evals[i], mse[i], psize[i]))

plt.figure(figsize=(10,6))
plt.plot(evals, mse)
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.EPS_0, xmin=0.0, xmax=1.0, color='r')
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Evaluations', fontsize=12)
plt.xlim([0,n*xcs.MAX_TRIALS])
plt.ylabel('Mean squared error', fontsize=12)
plt.show()

# show some predictions vs. answers
pred = xcs.predict(train_X)
print("*****************************")
print("first 10 predictions = ")
print(pred[:10])
print("*****************************")
print("first 10 answers = ")
print(train_Y[:10])
