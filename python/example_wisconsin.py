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
# This example uses the reinforcement learning mechanism to construct and update
# match and action sets with classifiers composed of hyperrectangle conditions,
# linear least squares predictions, and integer actions to solve UCI Wisconsin.
################################################################################

import xcsf.xcsf as xcsf # Import XCSF
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
np.set_printoptions(suppress=True)

##############################
# UCI Breast cancer wisconsin
##############################

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer() # 30 features, 2 classes # 569 instances
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
X_train = minmax_scale(X_train, feature_range=(0,1))
X_test = minmax_scale(X_test, feature_range=(0,1))
features = 30
classes = 2
train_len = len(X_train)
test_len = len(X_test)
print("train len = %d, test len = %d" % (train_len, test_len))

###################
# Initialise XCSF
###################

# initialise XCSF for single-step reinforcement learning
xcs = xcsf.XCS(features, classes, False)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 1000
xcs.PERF_TRIALS = 1000
xcs.EPS_0 = 0.01 # target error
xcs.COND_TYPE = 1 # hyperrectangles
xcs.PRED_TYPE = 1 # linear least squares
xcs.ACT_TYPE = 0 # integers

xcs.print_params()

#####################
# Execute experiment
#####################

n = 100 # 100,000 trials
trials = np.zeros(n)
psize = np.zeros(n)
msize = np.zeros(n)
performance = np.zeros(n)
error = np.zeros(n)
bar = tqdm(total=n) # progress bar

for i in range(n):
    for j in range(xcs.PERF_TRIALS):
        # explore trial
        sample = randint(0, train_len-1)
        state = X_train[sample]
        answer = y_train[sample]
        xcs.single_reset() # clear sets
        action = xcs.single_decision(state, True) # build mset, aset, pa, and select action
        if action == answer:
            reward = 1
        else:
            reward = 0
        xcs.single_update(reward) # update aset and potentially run EA
        # exploit trial
        sample = randint(0, test_len-1)
        state = X_test[sample]
        answer = y_test[sample]
        xcs.single_reset() # clear sets
        action = xcs.single_decision(state, False) # false signifies exploit mode
        if action == answer:
            reward = 1
        else:
            reward = 0
        performance[i] += reward
        error[i] += xcs.single_error(reward) # calculate system prediction error
    performance[i] /= float(xcs.PERF_TRIALS)
    error[i] /= float(xcs.PERF_TRIALS)
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # update status
    status = ("trials=%d performance=%.5f error=%.5f psize=%d msize=%.1f"
            % (trials[i], performance[i], error[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

##################################
# plot XCSF learning performance
##################################

plt.figure(figsize=(10,6))
plt.plot(trials, performance, label='Performance')
plt.plot(trials, error, label='System error')
plt.grid(linestyle='dotted', linewidth=1)
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.xlim([0,n*xcs.PERF_TRIALS])
plt.legend()
plt.show()

###################
# Confusion Matrix
###################

yActual = []
yPredicted = []
for i in range(test_len):
    state = X_test[i]
    answer = y_test[i]
    xcs.single_reset()
    action = xcs.single_decision(state, False)
    yActual.append(answer)
    yPredicted.append(action)

confusion_matrix_val = confusion_matrix(yActual, yPredicted)
labels = ['Benign', 'Malignant']
fig = plt.figure()
subplt = fig.add_subplot(111)
csubplt = subplt.matshow(confusion_matrix_val)
plt.title('XCSF Confusion Matrix:')
fig.colorbar(csubplt)
subplt.set_xticklabels([''] + labels)
subplt.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("XCSF accuracy: "+str(accuracy_score(yActual, yPredicted)))
print("XCSF f1: "+str(f1_score(yActual, yPredicted, average=None)))

############################
# Compare with alternatives
############################

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000)
mlp.fit(X_train, y_train)

yActual = []
yPredicted = []
for i in range(test_len):
    state = X_test[i]
    answer = y_test[i]
    action = mlp.predict(X_test[i].reshape(1,-1))
    yActual.append(answer)
    yPredicted.append(action)

print("MLP accuracy: "+str(accuracy_score(yActual, yPredicted)))
print("MLP f1: "+str(f1_score(yActual, yPredicted, average=None)))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

yActual = []
yPredicted = []
for i in range(test_len):
    state = X_test[i]
    answer = y_test[i]
    action = dtc.predict(X_test[i].reshape(1,-1))
    yActual.append(answer)
    yPredicted.append(action)

print("Decision tree accuracy: "+str(accuracy_score(yActual, yPredicted)))
print("Decision tree f1: "+str(f1_score(yActual, yPredicted, average=None)))
