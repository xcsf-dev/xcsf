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
import sys
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from tqdm import tqdm
np.set_printoptions(suppress=True)

##################################

# Boston house price data https://www.openml.org/d/531
#data = fetch_openml(data_id=531) # 13 features, 1 output regression, 501 instances

# MNIST data https://www.openml.org/d/554
#data = fetch_openml(data_id=554) # 784 features, 10 classes, 70,000 instances
#train_X, test_X, train_Y, test_Y = train_test_split(data.data, data.target, test_size=10000)

# USPS data https://www.openml.org/d/41082
#data = fetch_openml(data_id=41082) # 256 features, 10 classes, 9298 instances

# UCI One-hundred plant species leaves https://www.openml.org/d/1493
#data = fetch_openml(data_id=1493) # 64 features, 100 classes, 1599 instances

# UCI Mushroom https://www.openml.org/d/24 -- contains missing values (nan)
#data = fetch_openml(data_id=24) # 22 categorical features (116 if one-hot encode), 2 classes

# UCI Thyroid-allbp https://www.openml.org/d/40474
#data = fetch_openml(data_id=40474) # 26 features, 5 classes

# UCI Isolet (Isolated Letter Speech Recognition) https://www.openml.org/d/300
#data = fetch_openml(data_id=300) # 617 features

# UCI Breast cancer wisconsin
#from sklearn.datasets import load_breast_cancer
#data = load_breast_cancer() # 30 features, 2 classes

# UCI Vowel (speech) https://www.openml.org/d/307
data = fetch_openml(data_id=307) # 12 features, 11 classes, 990 instances

# UCI Hill valley https://www.openml.org/d/1479
#data = fetch_openml(data_id=1479) # classification, 100 features, 2 classes, 1212 instances

# UCI Nomao https://www.openml.org/d/1486
#data = fetch_openml(data_id=1486) # classification, 118 features, 2 classes, 34465 instances

# UCI thyroid-bp https://www.openml.org/d/40474
#data = fetch_openml(data_id=40474) # classification, 26 features, 5 classes, 2800 instances

# split into training and test sets
train_X, test_X, train_Y, test_Y = train_test_split(data.data, data.target, test_size=0.1)

##################################

# scale features [0,1]
train_X = minmax_scale(train_X, feature_range=(0,1))
test_X = minmax_scale(test_X, feature_range=(0,1))
 
# one hot encode labels
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
train_Y = onehot_encoder.fit_transform(train_Y.reshape(-1, 1))
test_Y = onehot_encoder.fit_transform(test_Y.reshape(-1, 1))

print("train_X shape = "+str(np.shape(train_X)))
print("test_X shape = "+str(np.shape(test_X)))
print("train_Y shape = "+str(np.shape(train_Y)))
print("test_Y shape = "+str(np.shape(test_Y)))

# get number of input and output variables
xvars = np.shape(train_X)[1]
yvars = xvars  #np.shape(train_Y)[1]

print("xvars = "+str(xvars) + " yvars = " + str(yvars))

# initialise XCSF
xcs = xcsf.XCS(xvars, yvars)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 1000
xcs.MAX_TRIALS = 1000 # number of trials per fit()
xcs.COND_TYPE = 2 # evolved neural network conditions
xcs.PRED_TYPE = 4 # sgd neural network predictors
xcs.HIDDEN_NEURON_ACTIVATION = 1# 9 # selu
xcs.NUM_HIDDEN_NEURONS = 11
xcs.ETA = 0.01 # 0.01 # 0.001 # 0.01
xcs.MIN_CON=0
xcs.MAX_CON=1
xcs.EPS_0 = 0.01
#xcs.THETA_GA=50000000
xcs.SAM_TYPE=0
xcs.SAM_NUM=1
#xcs.S_MUTATION=0
xcs.P_MUTATION=0
xcs.P_FUNC_MUTATION=0
xcs.LOSS_FUNC=1
xcs.GA_SUBSUMPTION=False
xcs.SET_SUBSUMPTION=False

##################################

n = 100
trials = np.zeros(n)
psize = np.zeros(n)
msize = np.zeros(n)
train_err = np.zeros(n)
bar = tqdm(total=n) # progress bar
for i in range(n):
    # train
    train_err[i] = xcs.fit(train_X, train_X, True) # True = shuffle
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # update status
    #status = ("evals=%d train_error=%.5f psize=%d msize=%.1f cond=%.1f pred=%.1f smut=%.3f pmut=%.3f pfmut=%.3f"
    status = ("%d %.5f %d %.1f %.1f %.1f %.3f %.3f %.3f"
            % (trials[i], train_err[i], psize[i], msize[i],
                xcs.pop_avg_cond_size(), xcs.pop_avg_pred_size(),
                xcs.pop_avg_mu(0), xcs.pop_avg_mu(1), xcs.pop_avg_mu(2)))
    print(status)
    sys.stdout.flush()

    xcs.save("test.bin")
    print("NT = "+str(xcs.THETA_SUB))
    xcs.load("test.bin")
    print("AT = "+str(xcs.THETA_SUB))
    exit(0)

    #bar.set_description(status)
    #bar.refresh()
    #bar.update(1)
bar.close()

pred = xcs.predict(test_X)
test_err = mean_squared_error(pred, test_X)
print(test_err)

##################################
# Classification
##################################
#print("XCSF")
#pred = xcs.predict(test_X) # soft max predictions
#pred = np.argmax(pred, axis=1) # select most likely class
#pred = onehot_encoder.fit_transform(pred.reshape(-1,1))
#print(classification_report(test_Y, pred))

## compare with MLP regressor
##mlp = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='sgd', learning_rate='constant', learning_rate_init=0.001, batch_size=1, max_iter=2)
#mlp.fit(train_X, train_Y)
#print(classification_report(test_Y, mlp_pred)) # 0.98 f1-score on MNIST

#print("**********")
#mlp = MLPRegressor(hidden_layer_sizes=(3, ), activation='logistic', solver='adam', learning_rate='constant', learning_rate_init=0.01, max_iter=100)
#mlp = MLPRegressor(hidden_layer_sizes=(3, ), activation='logistic', solver='sgd', learning_rate='constant', learning_rate_init=0.1, max_iter=100)
#mlp.fit(train_X, train_X)
#mlp_pred = mlp.predict(train_X)
#train_err = mean_squared_error(mlp_pred, train_X)
#print("MLP train")
#print(train_err)
#mlp_pred = mlp.predict(test_X)
#test_err = mean_squared_error(mlp_pred, test_X)
#print("MLP test")
#print(test_err)

##################################
# Plot performance
##################################
#plt.figure(figsize=(10,6))
#plt.plot(trials, train_err, label='Train Error')
##psize[:] = [x / xcs.POP_SIZE for x in psize] # scale for plotting
##msize[:] = [x / xcs.POP_SIZE for x in msize]
##plt.plot(trials, psize, label='Population macro-classifiers / P')
##plt.plot(trials, msize, label='Avg. match-set macro-classifiers / P')
#plt.grid(linestyle='dotted', linewidth=1)
#plt.axhline(y=xcs.EPS_0, xmin=0.0, xmax=1.0, linestyle='dashed', color='k')
#plt.title('XCSF Training Performance', fontsize=14)
#plt.xlabel('Trials', fontsize=12)
#plt.xlim([0,n*xcs.MAX_TRIALS])
#plt.legend()
#plt.show()
