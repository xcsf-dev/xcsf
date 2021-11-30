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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import xcsf

###############################
# Load training and test data
###############################

# Load USPS data from https://www.openml.org/d/41082
data = fetch_openml(data_id=41082) # 256 features, 10 classes, 9298 instances
INPUT_HEIGHT = 16
INPUT_WIDTH = 16
INPUT_CHANNELS = 1

# split into training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(data.data, data.target, test_size=0.1)

# scale features [0,1]
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# one hot encode labels
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
onehot_encoder.fit(y_train.reshape(-1, 1))
y_train = onehot_encoder.transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.transform(y_test.reshape(-1, 1))

# 10% of training for validation
X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.1)

print('X_train shape = ' + str(np.shape(X_train)))
print('y_train shape = ' + str(np.shape(y_train)))
print('X_val shape = ' + str(np.shape(X_val)))
print('y_val shape = ' + str(np.shape(y_val)))
print('X_test shape = ' + str(np.shape(X_test)))
print('y_test shape = ' + str(np.shape(y_test)))

# get number of input and output variables
X_DIM = np.shape(X_train)[1]
Y_DIM = np.shape(y_train)[1]
print('x_dim = ' + str(X_DIM) + ', y_dim = ' + str(Y_DIM))

###################
# Initialise XCSF
###################

xcs = xcsf.XCS(X_DIM, Y_DIM, 1) # initialise XCSF for supervised learning

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.LOSS_FUNC = 'onehot' # one-hot encoding classification error
xcs.E0 = 0.01 # 1% target error
xcs.ALPHA = 1
xcs.BETA = 0.05
xcs.NU = 5
xcs.THETA_EA = 100
xcs.THETA_DEL = 100
xcs.action('integer') # (dummy) integer actions

ACTIVATION = 'selu'
SGD_WEIGHTS = True
EVOLVE_WEIGHTS = True
EVOLVE_CONNECT = True
EVOLVE_ETA = True
EVOLVE_NEURONS = True
ETA = 0.01
ETA_MIN = 0.00001
MOMENTUM = 0.9
DECAY = 0
N_INIT = 5
N_MAX = 100
MAX_GROW = 1

condition_layers = {
    'layer_0': { # hidden layer
        'type': 'connected',
        'activation': ACTIVATION,
        'evolve-weights': EVOLVE_WEIGHTS,
        'evolve-connect': EVOLVE_CONNECT,
        'evolve-neurons': EVOLVE_NEURONS,
        'n-init': 1,
        'n-max': N_MAX,
        'max-neuron-grow': MAX_GROW,
    },
    'layer_1': { # output layer
        'type': 'connected',
        'activation': 'linear',
        'evolve-weights': EVOLVE_WEIGHTS,
        'evolve-connect': EVOLVE_CONNECT,
        'n-init': 1,
    }
}
xcs.condition('neural', condition_layers) # neural network conditions

layer_conv = {
    'type': 'convolutional',
    'activation': ACTIVATION,
    'sgd-weights': SGD_WEIGHTS,
    'evolve-weights': EVOLVE_WEIGHTS,
    'evolve-connect': EVOLVE_CONNECT,
    'evolve-eta': EVOLVE_ETA,
    'evolve-neurons': EVOLVE_NEURONS,
    'max-neuron-grow': MAX_GROW,
    'eta': ETA,
    'eta-min': ETA_MIN,
    'momentum': MOMENTUM,
    'decay': DECAY,
    'n-init': N_INIT,
    'n-max': N_MAX,
    'stride': 1,
    'size': 3,
    'pad': 1,
    'height': INPUT_HEIGHT,
    'width': INPUT_WIDTH,
    'channels': INPUT_CHANNELS,
}

layer_maxpool = {
    'type': 'maxpool',
    'stride': 2,
    'size': 2,
    'pad': 0,
    'height': INPUT_HEIGHT,
    'width': INPUT_WIDTH,
    'channels': INPUT_CHANNELS,
}

layer_connected = {
    'type': 'connected',
    'activation': ACTIVATION,
    'sgd-weights': SGD_WEIGHTS,
    'evolve-weights': EVOLVE_WEIGHTS,
    'evolve-connect': EVOLVE_CONNECT,
    'evolve-eta': EVOLVE_ETA,
    'evolve-neurons': EVOLVE_NEURONS,
    'max-neuron-grow': MAX_GROW,
    'eta': ETA,
    'eta-min': ETA_MIN,
    'momentum': MOMENTUM,
    'decay': DECAY,
    'n-init': N_INIT,
    'n-max': N_MAX,
}

prediction_layers = {
    'layer_0': layer_conv,
    'layer_1': layer_maxpool,
    'layer_2': layer_conv,
    'layer_3': layer_maxpool,
    'layer_4': layer_connected,
    'layer_out1': { # output layer - softmax composed of two layers
        'type': 'connected',
        'activation': 'linear',
        'sgd-weights': SGD_WEIGHTS,
        'evolve-weights': EVOLVE_WEIGHTS,
        'evolve-connect': EVOLVE_CONNECT,
        'evolve-eta': EVOLVE_ETA,
        'eta': ETA,
        'eta-min': ETA_MIN,
        'momentum': MOMENTUM,
        'decay': DECAY,
        'n-init': Y_DIM,
    },
    'layer_out2': { # output layer - softmax composed of two layers
        'type': 'softmax',
        'scale': 1,
    }
}
xcs.prediction('neural', prediction_layers) # neural network predictions

xcs.print_params()

##################################
# Run experiment
##################################

N = 100 # 100,000 trials
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
train_err = np.zeros(N)
val_err = np.zeros(N)
val_min = 1000 # minimum validation error observed
val_trial = 0 # number of trials at validation minimum

bar = tqdm(total=N) # progress bar
for i in range(N):
    # train
    train_err[i] = xcs.fit(X_train, y_train, True) # True = shuffle
    trials[i] = xcs.time() # number of learning trials so far
    psize[i] = xcs.pset_size() # current population size
    msize[i] = xcs.mset_size() # avg match set size
    # checkpoint lowest validation error
    val_err[i] = xcs.score(X_val, y_val, 1000) # use maximum of 1000 samples
    if val_err[i] < val_min:
        xcs.store()
        val_min = val_err[i]
        val_trial = trials[i]
    # update status
    status = ('trials=%d train_err=%.5f val_err=%.5f psize=%d msize=%.1f' %
              (trials[i], train_err[i], val_err[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# final XCSF test score
print('*****************************')
print('Restoring system from trial %d with val_err=%.5f' % (val_trial, val_min))
xcs.retrieve()
pred = xcs.predict(X_test) # soft max predictions
pred = np.argmax(pred, axis=1) # select most likely class
pred = onehot_encoder.fit_transform(pred.reshape(-1, 1))
inv_y_test = onehot_encoder.inverse_transform(y_test)
inv_pred = onehot_encoder.inverse_transform(pred)
print(classification_report(inv_y_test, inv_pred, digits=4))

# plot XCSF learning performance
plt.figure(figsize=(10, 6))
plt.plot(trials, train_err, label='Train Error')
plt.plot(trials, val_err, label='Validation Error')
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.E0, xmin=0, xmax=1, linestyle='dashed', color='k')
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.xlim([0, N * xcs.MAX_TRIALS])
plt.legend()
plt.show()
