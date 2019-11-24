# XCSF learning classifier system

An implementation of XCSF that can be built as either a stand-alone binary or as a Python library.

License|Linux Build|OSX Build|Fossa|Code Quality
:--:|:--:|:--:|:--:|:--:
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)|[![Linux Build Status](http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=linux&label=build&branch=master)](https://travis-ci.org/rpreen/xcsf)|[![OSX Build Status](http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=osx&label=build&branch=master)](https://travis-ci.org/rpreen/xcsf)|[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Frpreen%2Fxcsf.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Frpreen%2Fxcsf?ref=badge_shield)|[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2213b9ad4e034482bf058d4598d1618b)](https://www.codacy.com/app/rpreen/xcsf)[![Language grade: C/C++](https://img.shields.io/lgtm/grade/cpp/g/rpreen/xcsf.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rpreen/xcsf/context:cpp)

*******************************************************************************

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

### Evolved Actions

* `ACT_TYPE = 0`: Integer actions

### Computed Predictions

* `PRED_TYPE = 0`: Linear least squares
* `PRED_TYPE = 1`: Quadratic least squares
* `PRED_TYPE = 2`: Linear recursive least squares
* `PRED_TYPE = 3`: Quadratic recursive least squares
* `PRED_TYPE = 4`: Stochastic gradient descent multilayer perceptron neural networks

*******************************************************************************

## Compiler options

* `XCSF_PYLIB = ON`: Python library (CMake default = OFF)
* `PARALLEL = ON`: CPU parallelised matching, predicting, and updating with OpenMP (CMake default = ON)
* `GNUPLOT = ON`: real-time Gnuplot of the system error; data saved in folder: `out` (CMake default = OFF)
  
*******************************************************************************

## Requirements

### Stand-alone binary
 
* C11 compliant compiler.
* [CMake](https://www.cmake.org "CMake") (>= 3.12)
* [OpenMP](https://www.openmp.org "OpenMP") (Optional: PARALLEL=ON): supported by [GCC](https://gcc.gnu.org "GCC") and Clang with libomp.
* [Gnuplot](https://www.gnuplot.info "Gnuplot") (Optional: GNUPLOT=ON)

### Python library
 
* All of the above for building the stand-alone executable.
* C++11 compliant compiler.
* [Python](https://www.python.org "Python") (>= 3)
* [Boost](https://www.boost.org "Boost") (>= 1.56.0 for Python3)

*******************************************************************************

## Building

### Ubuntu 18.04

```
$ sudo apt install python3 libboost-python-dev libboost-numpy-dev gnuplot
$ git clone --recurse-submodules git@github.com:rpreen/xcsf.git
$ cd xcsf/build
$ cmake -DCMAKE_BUILD_TYPE=RELEASE -DXCSF_PYLIB=ON ..
$ make
```
 
### OSX (XCode 10.1 / Clang)

```
$ brew install libomp cmake python boost-python3 gnuplot
$ git clone --recurse-submodules git@github.com:rpreen/xcsf.git
$ cd xcsf/build
$ cmake -DCMAKE_BUILD_TYPE=RELEASE -DXCSF_PYLIB=ON ..
$ make
```

### Documentation (Doxygen + graphviz)

After running cmake:

```
$ make doc
```
 
*******************************************************************************

## Running

### Stand-alone

After building with CMake option: `-DXCSF_PYLIB=OFF`

There are currently 3 built-in problem environments: {csv, mp, maze}.

Example real-multiplexer classification:

```
$ ./xcsf/xcsf mp 6
```

Example discrete mazes:

```
$ ./xcsf/xcsf maze ../env/maze4.txt
```

Example regression with csv input: learning `data/sine_3var_train.csv` and testing `data/sine_3var_test.csv`

```
$ ./xcsf/xcsf csv ../data/sine_3var
```

### Python library

After building with CMake option: `-DXCSF_PYLIB=ON`

See example MyProject.py. Currently only regression is supported.

```
$ python3 MyProject.py
```
