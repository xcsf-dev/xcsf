# Python Library Usage

## Table of Contents

* [Constructor](#constructor)
* [Initialising General Parameters](#initialising-general-parameters)
* [Initialising Conditions](#initialising-conditions)
    * [Always match (dummy)](#always-match-dummy)
    * [Ternary Bitstrings](#ternary-bitstrings)
    * [Hyperrectangles and Hyperellipsoids](#hyperrectangles-and-hyperellipsoids)
    * [GP Trees](#gp-trees)
    * [DGP Graphs](#dgp-graphs)
    * [Neural Networks](#neural-networks)
* [Initialising Actions](#initialising-actions)
    * [Integers](#integers)
    * [Neural Networks](#neural-networks)
* [Initialising Predictions](#initialising-predictions)
    * [Constant](#constant)
    * [Normalised Least Mean Squares](#normalised-least-mean-squares)
    * [Recursive Least Mean Squares](#recursive-least-mean-squares)
    * [Neural Networks](#neural-networks)
* [Initialising Neural Networks](#neural-network-initialisation)
    * [General Network Specification](#general-network-specification)
    * [Activation Functions](#activation-functions)
    * [Connected Layers](#connected-layers)
    * [Recurrent Layers](#recurrent-layers)
    * [LSTM Layers](#lstm-layers)
    * [Softmax Layers](#softmax-layers)
    * [Dropout Layers](#dropout-layers)
    * [Noise Layers](#noise-layers)
    * [Convolutional Layers](#convolutional-layers)
    * [Max-pooling Layers](#max-pooling-layers)
    * [Average-pooling Layers](#average-pooling-layers)
    * [Upsampling Layers](#upsampling-layers)
* [Saving and Loading XCSF](#saving-and-loading-xcsf)
* [Storing and Retrieving XCSF](#storing-and-retrieving-xcsf)
* [Printing XCSF](#printing-xcsf)
* [XCSF Getters](#xcsf-getters)
* [Reinforcement Learning](#reinforcement-learning)
    * [Reinforcement Initialisation](#reinforcement-initialisation)
    * [Reinforcement Learning Method 1](#reinforcement-learning-method-1)
    * [Reinforcement Learning Method 2](#reinforcement-learning-method-2)
    * [Reinforcement Examples](#reinforcement-examples)
* [Supervised Learning](#supervised-learning)
    * [Supervised Initialisation](#supervised-initialisation)
    * [Supervised Fitting](#supervised-fitting)
    * [Supervised Scoring](#supervised-scoring)
    * [Supervised Predicting](#supervised-predicting)
    * [Supervised Examples](#supervised-examples)
* [Notes](#notes)

*******************************************************************************

## Constructor

XCSF can be initialised in the following way. The `x_dim` is an integer
specifying the number of input feature variables; `y_dim` is an integer
specifying the number of predicted target variables (1 for reinforcement
learning); and `n_actions` is an integer specifying the number of actions or
classes (1 for supervised learning). An optional fourth argument may be
supplied as a string specifying the location of an `.ini` configuration file.

```python
import xcsf.xcsf as xcsf

xcs = xcsf.XCS(x_dim, y_dim, n_actions)
```

*******************************************************************************

## Initialising General Parameters

Default parameter values are hard-coded within XCSF. The `.ini` configuration
file is parsed upon invocation of the [constructor](#constructor) and overrides
these values. At run-time, the values may be further overridden within python
by using the following properties:

```python
# General XCSF
xcs.OMP_NUM_THREADS = 8 # number of CPU cores to use 
xcs.POP_INIT = True # whether to seed the population with random rules
xcs.POP_SIZE = 200 # maximum population size
xcs.MAX_TRIALS = 1000 # number of trials to execute for each xcs.fit()
xcs.PERF_TRIALS = 1000 # number of trials to avg performance
xcs.LOSS_FUNC = 'mae' # mean absolute error
xcs.LOSS_FUNC = 'mse' # mean squared error
xcs.LOSS_FUNC = 'rmse' # root mean squared error
xcs.LOSS_FUNC = 'log' # log loss (cross-entropy)
xcs.LOSS_FUNC = 'binary-log' # binary log loss
xcs.LOSS_FUNC = 'onehot' # one-hot encoding classification error
xcs.LOSS_FUNC = 'huber' # Huber error
xcs.HUBER_DELTA = 1 # delta parameter for Huber error calculation

# General Classifier
xcs.E0 = 0.01 # target error, under which accuracy is set to 1
xcs.ALPHA = 0.1 # accuracy offset for rules above E0 (1=disabled)
xcs.NU = 5 # accuracy slope for rules with error above E0
xcs.BETA = 0.1 # learning rate for updating error, fitness, and set size
xcs.DELTA = 0.1 # fraction of least fit classifiers to increase deletion vote
xcs.THETA_DEL = 20 # min experience before fitness used in probability of deletion
xcs.INIT_FITNESS = 0.01 # initial classifier fitness
xcs.INIT_ERROR = 0 # initial classifier error
xcs.M_PROBATION = 10000 # trials since creation a rule must match at least 1 input or be deleted
xcs.STATEFUL = True # whether classifiers should retain state across trials
xcs.SET_SUBSUMPTION = False # whether to perform set subsumption
xcs.THETA_SUB = 100 # minimum experience of a classifier to become a subsumer

# Multi-step Problems
xcs.TELETRANSPORTATION = 50 # num steps to reset a multistep problem if goal not found
xcs.GAMMA = 0.95 # discount factor in calculating the reward for multistep problems
xcs.P_EXPLORE = 0.9 # probability of exploring vs. exploiting in a multistep trial

# Evolutionary Algorithm
xcs.EA_SELECT_TYPE = 'roulette' # roulette wheel parental selection
xcs.EA_SELECT_TYPE = 'tournament' # tournament parental selection
xcs.EA_SELECT_SIZE = 0.4 # fraction of set size for tournament parental selection
xcs.THETA_EA = 50 # average set time between EA invocations
xcs.LAMBDA = 2 # number of offspring to create each EA invocation
xcs.P_CROSSOVER = 0.8 # probability of applying crossover
xcs.ERR_REDUC = 1.0 # amount to reduce an offspring error (1=disabled)
xcs.FIT_REDUC = 0.1 # amount to reduce an offspring fitness (1=disabled)
xcs.EA_SUBSUMPTION = False # whether to try and subsume offspring classifiers
xcs.EA_PRED_RESET = False # whether to reset offspring predictions instead of copying
```

*******************************************************************************

## Initialising Conditions

### Always match (dummy)

```python
xcs.condition('dummy')
```

### Ternary Bitstrings

```python
args = {
    'bits': 2, # number of bits per float to binarise inputs
    'p-dontcare': 0.5, # don't care probability during covering
}
xcs.condition('ternary', args)
```

### Hyperrectangles and Hyperellipsoids

```python
args = {
    'min': 0, # minimum value of a center
    'max': 1, # maximum value of a center
    'min-spread': 0.1, # minimum initial spread
    'eta': 0, # gradient descent rate for moving centers to mean inputs matched
}
xcs.condition('hyperrectangle', args)
xcs.condition('hyperellipsoid', args)
```

### GP Trees

```python
args = {
    'min': 0, # minimum value of a constant
    'max': 1, # maximum value of a constant
    'n-constants': 100, # number of (global) constants available
    'init-depth': 5, # initial depth of a tree
    'max-len': 10000, # maximum initial length of a tree
}
xcs.condition('tree-gp', args)
```

### DGP Graphs

```python
args = {
    'max-k': 2, # number of connections per node
    'max-t': 10, # maximum number of cycles to update graphs
    'n': 20, # number of nodes in the graph
    'evolve-cycles': True, # whether to evolve the number of update cycles
}
xcs.condition('dgp', args)
xcs.condition('rule-dgp', args) # conditions + actions in single DGP graphs
```

### Neural Networks

Condition output layers should be ```'n-init': 1```.

Rule output layers should be ```'n-init': 1 + binary``` where binary is the
number of outputs required to output binary actions. For example, for 8 actions,
3 binary outputs are required and the output layer should be of size 4.

See [Neural Network Initialisation](#neural-network-initialisation).

```python
xcs.condition('neural', layer_args)
xcs.condition('rule-neural', layer_args) # conditions + actions in single neural nets
```

*******************************************************************************

## Initialising Actions

### Integers

```python
xcs.action('integer')
```

### Neural Networks

Output layer should be a softmax.
See [Neural Network Initialisation](#neural-network-initialisation).

```python
xcs.action('neural', layer_args)
```

*******************************************************************************

## Initialising Predictions

### Constant

```python
xcs.BETA = 0.1 # classifier update rate includes constant predictions
xcs.prediction('constant')
```

### Normalised Least Mean Squares

```python
args = {
    'x0': 1, # offset value
    'eta': 0.1, # gradient descent update rate (maximum value, if evolved)
    'eta-min': 0.0001, # minimum gradient descent update rate (if evolved)
    'evolve-eta': True, # whether to evolve the gradient descent rate
}
xcs.prediction('nlms-linear', args)
xcs.prediction('nlms-quadratic', args)
```

### Recursive Least Mean Squares

```python
args = {
    'x0': 1, # offset value
    'rls-scale-factor': 1000, # initial diagonal values of the gain-matrix
    'rls-lambda': 1, # forget rate (small values may be unstable)
}
xcs.prediction('rls-linear', args)
xcs.prediction('rls-quadratic', args)
```

### Neural Networks

Output layer should be ```'n-init': y_dim```.
See [Neural Network Initialisation](#neural-network-initialisation).

```python
xcs.prediction('neural', layer_args)
```

*******************************************************************************

## Neural Network Initialisation

### General Network Specification

```python
layer_args = {
    'layer_0': { # first hidden layer
        'type': 'connected', # layer type
        ..., # layer specific parameters
    },
    ..., # as many layers as desired
    'layer_n': { # output layer
        'type': 'connected', # layer type
        ..., # layer specific parameters
    },          
}
```

### Activation Functions

```python
'logistic', # logistic [0,1]
'relu', # rectified linear unit [0,inf]
'tanh', # tanh [-1,1]
'linear', # linear [-inf,inf]
'gaussian', # Gaussian (0,1]
'sin', # sine [-1,1]
'cos', # cosine [-1,1]
'softplus', # soft plus [0,inf]
'leaky', # leaky rectified linear unit [-inf,inf]
'selu', # scaled exponential linear unit [-1.7581,inf]
'loggy', # logistic [-1,1]
```

### Connected Layers

```python
layer_args = {
    'layer_0': {
        'type': 'connected', # layer type
        'activation': 'relu', # activation function
        'evolve-weights': True, # whether to evolve weights
        'evolve-connect': True, # whether to evolve connectivity
        'evolve-functions': True, # whether to evolve activation function
        'evolve-neurons': True, # whether to evolve the number of neurons
        'max-neuron-grow': 5, # maximum number of neurons to add or remove per mut
        'n-init': 10, # initial number of neurons
        'n-max': 100, # maximum number of neurons (if evolved)
        'sgd-weights': True, # whether to use gradient descent (only for predictions)
        'evolve-eta': True, # whether to evolve the gradient descent rate   
        'eta': 0.1, # gradient descent update rate (maximum value, if evolved)
        'eta-min': 0.0001, # minimum gradient descent update rate (if evolved)
        'momentum': 0.9, # momentum for gradient descent update
        'decay': 0, # weight decay during gradient descent update
    },       
}
```

### Recurrent Layers

```python
layer_args = {
    'layer_0': {
        'type': 'recurrent',
        ..., # other parameters same as for connected layers
    }
}
```

### LSTM Layers

```python
layer_args = {
    'layer_0': {
        'type': 'lstm',
        'activation': 'tanh', # activation function
        'recurrent-activation': 'logistic', # recurrent activation function
        ..., # other parameters same as for connected layers
    }
}
```

### Softmax Layers

Softmax layers can be composed of a linear connected layer and softmax:

```python
layer_args = {
    'layer_0': {
        'type': 'connected',
        'n-init': N_ACTIONS, # number of (softmax) outputs
        ..., # other parameters same as for connected layers
    },       
    'layer_1': {
        'type': 'softmax',
        'scale': 1, # softmax temperature
    },       
}
```

### Dropout Layers

```python
layer_args = {
    'layer_0': {
        'type': 'dropout',
        'probability': 0.2, # probability of dropping an input
    }
}
```

### Noise Layers

Gaussian noise adding layers.

```python
layer_args = {
    'layer_0': {
        'type': 'noise',
        'probability': 0.2, # probability of adding noise to an input
        'scale': 1.0, # standard deviation of Gaussian noise added
    }
}
```

### Convolutional Layers

Convolutional layers require image inputs and produce image outputs. If used as
the first layer, the width, height, and number of channels must be specified.
If ```'evolve-neurons': True``` the number of filters will be evolved using an
initial number of filters ```'n-init'``` and maximum number ```'n-max'```.

```python
layer_args = {
    'layer_0': {
        'type': 'convolutional',
        'activation': 'relu', # activation function
        'height': 16, # input height
        'width': 16, # input width
        'channels': 1, # number of input channels
        'n-init': 6, # number of convolutional kernel filters
        'size': 3, # the size of the convolution window
        'stride': 1, # the stride of the convolution window
        'pad': 1, # the padding of the convolution window
        ..., # other parameters same as for connected layers
    },       
    'layer_1': {
        'type': 'convolutional',
        ..., # parameters same as above; height, width, channels not needed
    },       
}
```

### Max-pooling Layers

Max-pooling layers require image inputs and produce image outputs. If used as
the first layer, the width, height, and number of channels must be specified.

```python
layer_args = {
    'layer_0': {
        'type': 'maxpool',
        'height': 16, # input height
        'width': 16, # input width
        'channels': 1, # number of input channels
        'size': 2, # the size of the maxpooling operation
        'stride': 2, # the stride of the maxpooling operation
        'pad': 0, # the padding of the maxpooling operation
    },       
    'layer_1': {
        'type': 'maxpool',
        'size': 2,
        'stride': 2,
        'pad': 0,
    },       
}
```

### Average-pooling Layers

Average-pooling layers require image inputs. If used as the first layer, the
width, height, and number of channels must be specified. Outputs an average for
each input channel.

```python
layer_args = {
    'layer_0': {
        'type': 'avgpool',
        'height': 16, # input height
        'width': 16, # input width
        'channels': 1, # number of input channels
    },       
    'layer_1': {
        'type': 'avgpool',
    },       
}
```

### Upsampling Layers

Upsampling layers require image inputs and produce image outputs. If used as
the first layer, the width, height, and number of channels must be specified.

```python
layer_args = {
    'layer_0': {
        'type': 'upsample',
        'height': 16, # input height
        'width': 16, # input width
        'channels': 1, # number of input channels
        'stride': 2, # the stride of the upsampling operation
    },       
    'layer_1': {
        'type': 'upsample',
        'stride': 2,
    },       
}
```

*******************************************************************************

## Saving and Loading XCSF

To save the entire current state of XCSF to a binary file:

```python
xcs.save('saved_name.bin')
```

To load the entire state of XCSF from a binary file:

```python
xcs.load('saved_name.bin')
```

*******************************************************************************

## Storing and Retrieving XCSF

To store the current XCSF population in memory for later retrieval, overwriting
any previously stored population:

```python
xcs.store()
```

To retrieve the previously stored XCSF population from memory:

```python
xcs.retrieve()
```

*******************************************************************************

## Printing XCSF

To print the current XCSF parameters:

```python
xcs.print_params()
```

To print the current XCSF population:

```python
print_condition = True # whether to print the classifier conditions
print_action = True # whether to print the classifier actions
print_prediction = True # whether to print the classifier predictions
xcs.print_pset(print_condition, print_action, print_prediction)
```

*******************************************************************************

## XCSF Getters

Values for all [general parameters](#initialising-general-parameters) are
directly accessible via the property. Specific getter functions:

```python
# General
xcs.pset_size() # returns the mean population size
xcs.pset_num() # returns the mean population numerosity
xcs.mset_size() # returns the mean match set size
xcs.aset_size() # returns the mean action set size
xcs.mfrac() # returns the mean fraction of inputs matched by the best rule
xcs.time() # returns the current EA time
xcs.version_major() # returns the XCSF major version number
xcs.version_minor() # returns the XCSF minor version number
xcs.version_build() # returns the XCSF build version number
xcs.pset_mean_cond_size() # returns the mean condition size
xcs.pset_mean_pred_size() # returns the mean prediction size

# Neural network specific - population set averages
# 'layer' argument is an integer specifying the location of a layer: first layer=0
xcs.pset_mean_pred_eta(layer) # returns the mean eta for a prediction layer
xcs.pset_mean_pred_neurons(layer) # returns the mean number of neurons for a prediction layer
xcs.pset_mean_pred_layers() # returns the mean number of layers in the prediction networks
xcs.pset_mean_pred_connections(layer) # returns the number of active connections for a prediction layer
xcs.pset_mean_cond_neurons(layer) # returns the mean number of neurons for a condition layer
xcs.pset_mean_cond_layers() # returns the mean number of layers in the condition networks
xcs.pset_mean_cond_connections(layer) # returns the number of active connections for a condition layer
```

*******************************************************************************

## Reinforcement Learning

### Reinforcement Initialisation

Initialise XCSF with `y_dim = 1` for predictions to estimate the scalar reward.

```python
import xcsf.xcsf as xcsf
xcs = xcsf.XCS(x_dim, 1, n_actions)
```

### Reinforcement Learning Method 1

The standard method involves the basic loop as shown below. `state` must be a
1-D numpy array representing the feature values of a single instance; `reward`
must be a scalar value representing the current environmental reward for having
performed the action; and `done` must be a boolean value representing whether
the environment is currently in a terminal state.

```python
state = env.reset()
xcs.init_trial()
for cnt in range(xcs.TELETRANSPORTATION):
    xcs.init_step()
    action = xcs.decision(state, explore) # explore specifies whether to explore/exploit
    next_state, reward, done = env.step(action)
    xcs.update(reward, done) # update the current action set and/or previous action set
    err += xcs.error(reward, done, env.max_payoff()) # system prediction error
    xcs.end_step()
    if done:
        break
    state = next_state
cnt += 1
xcs.end_trial()
```

### Reinforcement Learning Method 2

The `fit()` function may be used as below to execute one single-step learning
trial, i.e., creation of the match and action sets, updating the action set and
running the EA as appropriate. The vector `state` must be a 1-D numpy array
representing the feature values of a single instance; `action` must be an
integer representing the selected action (and therefore the action set to
update); and `reward` must be a scalar value representing the current
environmental reward for having performed the action.

```python
xcs.fit(state, action, reward)
```

The entire prediction array for a given state can be returned using the
supervised `predict()` function, which must receive a 2-D numpy array. For
example:

```python
prediction_array = xcs.predict(state.reshape(1,-1))[0]
```

### Reinforcement Examples

Reinforcement learning examples with action sets:

`example_rmux.py`
`example_maze.py`

Reinforcement learning example (using experience replay) without action sets:

`example_cartpole.py`

*******************************************************************************

## Supervised Learning

### Supervised Initialisation

Initialise XCSF with a single (dummy) integer action. Set
[conditions](#initialising-conditions) and
[predictions](#initialising-predictions) as desired.

```python
import xcsf.xcsf as xcsf
xcs = xcsf.XCS(x_dim, y_dim, 1) # single action
xcs.action('integer') # dummy integer actions
```

### Supervised Fitting

The `fit()` function may be used as below to execute `xcs.MAX_TRIALS` number of
learning iterations (i.e., single-step trials) using a supplied training set.
The input arrays `X_train` and `y_train` must be 2-D numpy arrays. The third
parameter specifies whether to randomly shuffle the training data. The function
will return the training prediction error using the loss function as specified
by `xcs.LOSS_FUNC`. Returns a scalar representing the error.

```python
train_error = xcs.fit(X_train, y_train, True)
```

### Supervised Scoring

The `score()` function may be used as below to calculate the prediction error
over a single pass of a supplied data set without updates or the EA being
invoked (e.g., for scoring a validation set). An optional third argument may be
supplied that specifies the maximum number of iterations performed; if this
value is less than the number of instances supplied, samples will be drawn
randomly. Returns a scalar representing the error.

```python
val_error = xcs.score(X_val, y_val)
```

### Supervised Predicting

The `predict()` function may be used as below to calculate the XCSF predictions
for a supplied data set. No updates or EA invocations are performed. The input
vector must be a 2-D numpy array. Returns a 2-D numpy array.

```python
predictions = xcs.predict(X_test)
```

### Supervised Examples

`example_regression.py`
`example_classification.py`

*******************************************************************************

## Notes

The use of [always matching dummy conditions](#always-match-dummy) results in
the match set being equal to the population set, i.e., [M] = [P]. The
evolutionary algorithm and classifier updates are thus performed within [P],
and global models are designed (e.g., neural networks) that cover the entire
state-space. This configuration operates as a more traditional evolutionary
algorithm, which can be useful for debugging and benchmarking.

Additionally, a single global model (e.g., a linear regression) can be fit by
also setting `POP_SIZE = 1` and disabling the evolutionary algorithm by setting
the invocation frequency to a larger number than will ever be executed, e.g.,
`THETA_EA = 5000000`. This can also be useful for debugging and benchmarking.
