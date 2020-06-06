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

################################################################################
# This example uses the supervised learning mechanism to construct and update
# match sets with classifiers composed of neural network conditions and
# predictions similar to XCSF for function approximation. Classifier predictions
# use a softmax layer for output and labels are one-hot encoded.
################################################################################

import xcsf.xcsf as xcsf # Import XCSF
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
np.set_printoptions(suppress=True)

###############################
# Load training and test data
###############################

# Load USPS data from https://www.openml.org/d/41082
data = fetch_openml(data_id=41082) # 256 features, 10 classes, 9298 instances

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)

# normalise features (zero mean and unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# one hot encode labels
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))

print("X_train shape = "+str(np.shape(X_train)))
print("y_train shape = "+str(np.shape(y_train)))
print("X_test shape = "+str(np.shape(X_test)))
print("y_test shape = "+str(np.shape(y_test)))

# get number of input and output variables
x_dim = np.shape(X_train)[1]
y_dim = np.shape(y_train)[1]
print("x_dim = "+str(x_dim) + ", y_dim = " + str(y_dim))

###################
# Initialise XCSF
###################

# initialise XCSF
xcs = xcsf.XCS(x_dim, y_dim)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.LOSS_FUNC = 5 # one-hot encoding accuracy
xcs.EPS_0 = 0.01 # 1% target error
xcs.ALPHA = 0.1 # accuracy offset
xcs.NU = 5 # accuracy slope

xcs.COND_TYPE = 3 # neural network conditions
xcs.COND_OUTPUT_ACTIVATION = 3 # linear
xcs.COND_HIDDEN_ACTIVATION = 1 # relu
xcs.COND_NUM_NEURONS = [10] # initial neurons
xcs.COND_MAX_NEURONS = [100] # maximum neurons
xcs.COND_EVOLVE_WEIGHTS = True
xcs.COND_EVOLVE_NEURONS = True
xcs.COND_EVOLVE_FUNCTIONS = False

xcs.PRED_TYPE = 5 # neural network predictors
xcs.PRED_OUTPUT_ACTIVATION = 100 # soft max
xcs.PRED_HIDDEN_ACTIVATION = 1 # relu
xcs.PRED_NUM_NEURONS = [50] # initial neurons
xcs.PRED_MAX_NEURONS = [100] # maximum neurons
xcs.PRED_EVOLVE_WEIGHTS = True
xcs.PRED_EVOLVE_NEURONS = True
xcs.PRED_EVOLVE_FUNCTIONS = False
xcs.PRED_EVOLVE_ETA = True
xcs.PRED_SGD_WEIGHTS = True

xcs.print_params()

##################################
# Example plotting in matplotlib
##################################

n = 100 # 100,000 trials
trials = np.zeros(n)
psize = np.zeros(n)
msize = np.zeros(n)
train_err = np.zeros(n)
test_err = np.zeros(n)
bar = tqdm(total=n) # progress bar
for i in range(n):
    # train
    train_err[i] = xcs.fit(X_train, y_train, True) # True = shuffle
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # test
    test_err[i] = xcs.score(X_test, y_test)
    # update status
    status = ("trials=%d train_err=%.5f test_err=%.5f psize=%d msize=%.1f"
        % (trials[i], train_err[i], test_err[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# plot XCSF learning performance
plt.figure(figsize=(10,6))
plt.plot(trials, train_err, label='Train Error')
plt.plot(trials, test_err, label='Test Error')
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=xcs.EPS_0, xmin=0.0, xmax=1.0, linestyle='dashed', color='k')
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.xlim([0,n*xcs.MAX_TRIALS])
plt.legend()
plt.show()

# final XCSF test score
print("XCSF")
pred = xcs.predict(X_test) # soft max predictions
pred = np.argmax(pred, axis=1) # select most likely class
pred = onehot_encoder.fit_transform(pred.reshape(-1,1))
inv_y_test = onehot_encoder.inverse_transform(y_test)
inv_pred = onehot_encoder.inverse_transform(pred)
print(classification_report(inv_y_test, inv_pred, digits=4))
