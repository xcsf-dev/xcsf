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

"""
This example demonstrates the XCSF supervised learning mechanisms to perform
classification on the USPS handwritten digits dataset. Classifiers are composed
of neural network conditions and predictions. A softmax layer is used as
prediction output and labels are one-hot encoded. Similar to regression, a
single dummy action is performed such that [A] = [M].
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import minmax_scale, OneHotEncoder
import xcsf.xcsf as xcsf

###############################
# Load training and test data
###############################

# Load USPS data from https://www.openml.org/d/41082
data = fetch_openml(data_id=41082) # 256 features, 10 classes, 9298 instances

# split into training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(data.data, data.target, test_size=0.1)

# scale features [0,1]
X_train = minmax_scale(X_train, feature_range=(0, 1))
X_test = minmax_scale(X_test, feature_range=(0, 1))

# one hot encode labels
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))

print('X_train shape = ' + str(np.shape(X_train)))
print('y_train shape = ' + str(np.shape(y_train)))
print('X_test shape = ' + str(np.shape(X_test)))
print('y_test shape = ' + str(np.shape(y_test)))

# get number of input and output variables
X_DIM = np.shape(X_train)[1]
Y_DIM = np.shape(y_train)[1]
print('x_dim = ' + str(X_DIM) + ', y_dim = ' + str(Y_DIM))

###################
# Initialise XCSF
###################

# initialise XCSF for supervised learning
xcs = xcsf.XCS(X_DIM, Y_DIM, 1)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.LOSS_FUNC = 5 # one-hot encoding accuracy
xcs.EPS_0 = 0.01 # 1% target error
xcs.ALPHA = 0.1 # accuracy offset
xcs.NU = 5 # accuracy slope
xcs.action('integer') # (dummy) integer actions

cond_args = {
        'output-activation': 'linear',
        'hidden-activation': 'selu',
        'evolve-weights': True,
        'evolve-neurons': True,
        'evolve-functions': False,
        'evolve-connectivity': True,
        'num-neurons': [20], # number of initial neurons
        'max-neurons': [100], # maximum number of neurons
        'max-neuron-grow': 5, # max neurons to add/remove per mutation event
    }
xcs.condition('neural', cond_args)

pred_args = {
        'output-activation': 'softmax',
        'hidden-activation': 'selu',
        'sgd-weights': True,
        'eta': 0.001, # maximum gradient descent rate
        'momentum': 0.9,
        'decay': 0,
        'evolve-eta': True,
        'evolve-weights': True,
        'evolve-neurons': True,
        'evolve-functions': False,
        'evolve-connectivity': True,
        'num-neurons': [20], # number of initial neurons
        'max-neurons': [10], # maximum number of neurons
        'max-neuron-grow': 5, # max neurons to add/remove per mutation event
    }
xcs.prediction('neural', pred_args)

xcs.print_params()

##################################
# Example plotting in matplotlib
##################################

N = 100 # 100,000 trials
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
train_err = np.zeros(N)
test_err = np.zeros(N)
bar = tqdm(total=N) # progress bar
for i in range(N):
    # train
    train_err[i] = xcs.fit(X_train, y_train, True) # True = shuffle
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # test
    test_err[i] = xcs.score(X_test, y_test)
    # update status
    status = ('trials=%d train_err=%.5f test_err=%.5f psize=%d msize=%.1f' %
              (trials[i], train_err[i], test_err[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# plot XCSF learning performance
plt.figure(figsize=(10, 6))
plt.plot(trials, train_err, label='Train Error')
plt.plot(trials, test_err, label='Test Error')
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.EPS_0, xmin=0, xmax=1, linestyle='dashed', color='k')
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.xlim([0, N * xcs.MAX_TRIALS])
plt.legend()
plt.show()

# final XCSF test score
print('XCSF')
pred = xcs.predict(X_test) # soft max predictions
pred = np.argmax(pred, axis=1) # select most likely class
pred = onehot_encoder.fit_transform(pred.reshape(-1, 1))
inv_y_test = onehot_encoder.inverse_transform(y_test)
inv_pred = onehot_encoder.inverse_transform(pred)
print(classification_report(inv_y_test, inv_pred, digits=4))
