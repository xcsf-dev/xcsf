import xcsf.xcsf as xcsf
import random
import math
import numpy as np
import csv

# read csv data
def read_input_csv(path, name, t, v):
    tmp = []
    with open(path+name+"_"+t+"_"+v+".csv", 'rb') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            tmp.append(row)
    return tmp

data_path = "../data/"
data_name = "sine_1var"
train_X_list = read_input_csv(data_path, data_name, "train", "x")
train_Y_list = read_input_csv(data_path, data_name, "train", "y")
test_X_list = read_input_csv(data_path, data_name, "test", "x")
test_Y_list = read_input_csv(data_path, data_name, "test", "y")

# convert to numpy arrays
train_X = np.asarray(train_X_list, dtype=np.float64)
train_Y = np.asarray(train_Y_list, dtype=np.float64)
test_X = np.asarray(test_X_list, dtype=np.float64)
test_Y = np.asarray(test_Y_list, dtype=np.float64)

print("train_X = "+str(np.shape(train_X))+" train_Y = "+str(np.shape(train_Y)))
print("test_X = "+str(np.shape(test_X))+" test_Y = "+str(np.shape(test_Y)))      

num_inputs = np.shape(train_X)[1]
num_outputs = np.shape(train_Y)[1]

## initialise XCSF
xcs = xcsf.XCS(num_inputs, num_outputs)

# override cons.txt
xcs.MAX_TRIALS = 50000
xcs.PRED_TYPE = 4 # neural network predictors

# fit function
xcs.fit(train_X, train_Y, test_X, test_Y)
