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
train_X, test_X, train_Y, test_Y = train_test_split(data.data, data.target, test_size=0.2)
train_X = minmax_scale(train_X, feature_range=(0,1))
test_X = minmax_scale(test_X, feature_range=(0,1))
features = 30
classes = 2
train_len = len(train_X)
test_len = len(test_X)
print("train len = %d, test len = %d" % (train_len, test_len))

###################
# Initialise XCSF
###################

# initialise XCSF for single-step reinforcement learning
xcs = xcsf.XCS(features, classes, False)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.PERF_AVG_TRIALS = 1000
xcs.EPS_0 = 0.01 # target error
xcs.COND_TYPE = 0 # hyperrectangles
xcs.PRED_TYPE = 0 # linear least squares
xcs.ACT_TYPE = 0 # integers

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
    perf = 0
    err = 0
    for j in range(xcs.PERF_AVG_TRIALS):
        # explore trial
        sample = randint(0, train_len-1)
        state = train_X[sample]
        answer = train_Y[sample]
        xcs.single_reset() # clear sets
        action = xcs.single_decision(state, True) # build mset, aset, pa, and select action
        if action == answer:
            reward = 1
        else:
            reward = 0
        xcs.single_update(reward) # update aset and potentially run EA
        # exploit trial
        sample = randint(0, test_len-1)
        state = test_X[sample]
        answer = test_Y[sample]
        xcs.single_reset() # clear sets
        action = xcs.single_decision(state, False) # false signifies exploit mode
        if action == answer:
            reward = 1
        else:
            reward = 0
        perf += reward
        err += xcs.single_error(reward) # calculate system prediction error
    performance[i] = perf / xcs.PERF_AVG_TRIALS
    error[i] = err / xcs.PERF_AVG_TRIALS
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # update status
    status = ("trials=%d performance=%.5f error=%.5f psize=%d msize=%.1f smut=%.4f pmut=%.4f emut=%.4f fmut=%.4f"
            % (trials[i], performance[i], error[i], psize[i], msize[i],
                xcs.pop_avg_mu(0), xcs.pop_avg_mu(1), xcs.pop_avg_mu(2), xcs.pop_avg_mu(3)))
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
plt.xlim([0,n*xcs.PERF_AVG_TRIALS])
plt.legend()
plt.show()

###################
# Confusion Matrix
###################

yActual = []
yPredicted = []
for i in range(test_len):
    state = test_X[i]
    answer = test_Y[i]
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
mlp.fit(train_X, train_Y)

yActual = []
yPredicted = []
for i in range(test_len):
    state = test_X[i]
    answer = test_Y[i]
    action = mlp.predict(test_X[i].reshape(1,-1))
    yActual.append(answer)
    yPredicted.append(action)

print("MLP accuracy: "+str(accuracy_score(yActual, yPredicted)))
print("MLP f1: "+str(f1_score(yActual, yPredicted, average=None)))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(train_X, train_Y)

yActual = []
yPredicted = []
for i in range(test_len):
    state = test_X[i]
    answer = test_Y[i]
    action = dtc.predict(test_X[i].reshape(1,-1))
    yActual.append(answer)
    yPredicted.append(action)

print("Decision tree accuracy: "+str(accuracy_score(yActual, yPredicted)))
print("Decision tree f1: "+str(f1_score(yActual, yPredicted, average=None)))
