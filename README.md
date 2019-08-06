XCSF learning classifier system
------------------------

An implementation of XCSF that can be built either as a stand-alone binary or
as a Python library.

Features:
------------------------

Evolved Conditions:

COND_TYPE =-1: always matching dummy condition
COND_TYPE = 0: real-valued hyperrectangle intervals
COND_TYPE = 1: multilayer perceptron neural networks
COND_TYPE = 2: GP trees
COND_TYPE = 3: dynamical GP graphs

COND_TYPE = 11: both conditions and predictions in single dynamical GP graphs
COND_TYPE = 12: both conditions and predictions in single neural networks

Computed Predictions:

PRED_TYPE = 0: linear least squares
PRED_TYPE = 1: quadratic least squares
PRED_TYPE = 2: linear recursive least squares
PRED_TYPE = 3: quadratic recursive least squares
PRED_TYPE = 4: backpropagation multilayer perceptron neural networks
 
Mutation for conditions:
NUM_SAM = 0: fixed rates (P_MUTATION and S_MUTATION)
NUM_SAM > 0: self-adaptive rate
 
Compiler options:
------------------------

A real-time GNUPlot of the system error can be enabled by compiling with
GNUPLOT = ON

The matching and set prediction functions (where most processing occurs) can be
parallelised using OpenMP by compiling with PARALLEL = ON.
  
Stand-alone executable:
------------------------

Building:

0. Clone: `git clone --recurse-submodules git@github.com:rpreen/xcsf.git`
1. Change to the build directory: `cd xcsf/build`
2. Run cmake: `cmake .. -DCMAKE_BUILD_TYPE=RELEASE`
3. Run make: `make`

Running:

Example learning on `data/sine_1var_train` and testing on `data/sine_1var_test`:

1. Change to the application folder: `cd xcsf`
2. Modify constants as needed: `cons.txt`
3. Run: `./xcsf ../../data/sine_1var`              

Python library:
------------------------

Building:

0. Clone: `git clone --recurse-submodules git@github.com:rpreen/xcsf.git`
1. Change to the build directory: `cd xcsf/build`
2. Run cmake: `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DXCSF_PYTHON_LIBRARY`
3. Run make: `make`

Running:

Example learning on `data/sine_1var_train` and testing on `data/sine_1var_test`:

1. python MyProject.py
