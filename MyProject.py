#!/usr/bin/python3
import xcsf.xcsf as xcsf
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import minmax_scale

# load example data set
#data = datasets.load_iris() # classification
#data = datasets.load_digits() # classification
#data = datasets.load_wine() # classification
#data = datasets.load_breast_cancer() # classification
#data = datasets.load_linnerud() # multivariate regression
#data = datasets.load_diabetes() # regression
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
xcs.MAX_TRIALS = 50000
xcs.COND_TYPE = 0
xcs.PRED_TYPE = 0
 
# fit function
xcs.fit(train_X, train_Y)
