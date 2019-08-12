# XCSF learning classifier system

An implementation of XCSF for regression problems that can be built as either a
stand-alone binary or as a Python library.

License|Codacy Review|Linux Build|OSX Build|Fossa
:--:|:--:|:--:|:--:|:--:
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)|[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2213b9ad4e034482bf058d4598d1618b)](https://www.codacy.com/app/rpreen/xcsf)|[![Linux Build Status](http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=linux&label=build&branch=master)](https://travis-ci.org/rpreen/xcsf)|[![OSX Build Status](http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=osx&label=build&branch=master)](https://travis-ci.org/rpreen/xcsf)|[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Frpreen%2Fxcsf.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Frpreen%2Fxcsf?ref=badge_shield)

------------------------
## Features

See `default.ini` for a full list of options.

### Evolved Conditions

* `COND_TYPE =-1`: Always matching dummy condition
* `COND_TYPE = 0`: Hyperrectangles
* `COND_TYPE = 1`: Hyperellipsoids
* `COND_TYPE = 2`: Multilayer perceptron neural networks
* `COND_TYPE = 3`: GP trees
* `COND_TYPE = 4`: Dynamical GP graphs
* `COND_TYPE = 11`: Both conditions and predictions in single dynamical GP graphs
* `COND_TYPE = 12`: Both conditions and predictions in single neural networks

### Computed Predictions

* `PRED_TYPE = 0`: Linear least squares
* `PRED_TYPE = 1`: Quadratic least squares
* `PRED_TYPE = 2`: Linear recursive least squares
* `PRED_TYPE = 3`: Quadratic recursive least squares
* `PRED_TYPE = 4`: Stochastic gradient descent multilayer perceptron neural networks
  * `HIDDEN_NEURON_ACTIVATION = 0`: Logistic (-1,1)
  * `HIDDEN_NEURON_ACTIVATION = 1`: Rectified linear unit (0,inf)
  * `HIDDEN_NEURON_ACTIVATION = 2`: Gaussian (0,1)
  * `HIDDEN_NEURON_ACTIVATION = 3`: Bent identity (-inf,inf)
  * `HIDDEN_NEURON_ACTIVATION = 4`: TanH (-1,1)
  * `HIDDEN_NEURON_ACTIVATION = 5`: Sinusoid (-1,1)
  * `HIDDEN_NEURON_ACTIVATION = 6`: Soft plus (0,inf)
  * `HIDDEN_NEURON_ACTIVATION = 7`: Identity (-inf,inf)

 
### Mutation for conditions

* `NUM_SAM = 0`: Fixed rates for `P_MUTATION` and `S_MUTATION`
* `NUM_SAM = 1`: Self-adapts `P_MUTATION`
* `NUM_SAM = 2`: Self-adapts `P_MUTATION` and `S_MUTATION`
 
------------------------
## Compiler options

* `GNUPLOT = ON`: real-time GNUPlot of the system error; data saved in folder: `out`
* `PARALLEL = ON`: matching and set prediction functions parallelised with OpenMP
  
------------------------
## Requirements

### Stand-alone binary
 
* ![C11](https://img.shields.io/badge/C-11-blue.svg?style=flat) compliant compiler.
* The [cmake][cmake] build system: at least version 3.12.
* OpenMP (PARALLEL=ON): supported by the GNU Compiler Collections
* GNUPlot (GNUPLOT=ON)

### Python library
 
* All of the above for building the stand-alone executable.
* ![C++11](https://img.shields.io/badge/C++-11-blue.svg?style=flat) compliant compiler.
* Python
* Boost Python and numpy libraries (at least version 1.56.0 for Python3)

### Installing

* Ubuntu 18.04:
  * `sudo apt install python3 libboost-python-dev libboost-numpy-dev gnuplot`
* OS X:
  * `brew install cmake gcc libomp python3 boost-python3 gnuplot`
  
## Building

0. Clone: `git clone --recurse-submodules git@github.com:rpreen/xcsf.git`
1. Change to the build directory: `cd xcsf/build`
2. Run cmake:
	* Ubuntu 18.04: `cmake -DCMAKE_BUILD_TYPE=RELEASE -DXCSF_PYTHON_LIBRARY=ON ..`
	* OSX:  `cmake -DCMAKE_BUILD_TYPE=RELEASE -DXCSF_PYTHON_LIBRARY=ON -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 ..`
3. Run make: `make`

## Running

### Stand-alone

Arguments: 

1: (required) a path to input csv files.
2: (optional) a configuration file; defaults to using `default.ini`

Example learning on `data/sine_1var_train` and testing on `data/sine_1var_test`

Run: `./xcsf/xcsf ../data/sine_1var`              

### Python library

See example MyProject.py

Run: `python3 MyProject.py`

[cmake]: http://www.cmake.org/ "CMake tool"
