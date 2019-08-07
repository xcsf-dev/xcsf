#!/usr/bin/python3
import xcsf.xcsf as xcsf
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize

iris = datasets.load_iris()
train_X, train_Y = iris.data, iris.target

train_X = normalize(train_X)
train_Y = normalize(train_Y.reshape(train_Y.shape[0],-1), norm='max', axis=0)

print("train_X shape = "+str(np.shape(train_X)))
print("train_Y shape = "+str(np.shape(train_Y)))

xvars = np.shape(train_X)[1]
yvars = np.shape(train_Y)[1] 

print("xvars = "+str(xvars) + " yvars = " + str(yvars))

# initialise XCSF
xcs = xcsf.XCS(xvars, yvars)

# override cons.txt
xcs.MAX_TRIALS = 10000
xcs.COND_TYPE = 0
xcs.PRED_TYPE = 0
 
# fit function
xcs.fit(train_X, train_Y)
