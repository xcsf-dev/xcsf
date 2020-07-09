# XCSF learning classifier system

An implementation of the XCSF [learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) that can be built as a stand-alone binary or as a Python library.

*******************************************************************************

<table>
    <tr>
        <th>License</th>
        <th>Linux Build</th>
        <th>OSX Build</th>
        <th>Windows Build</th>
    </tr>
    <tr>
        <td><a href="http://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg"></a></td>
        <td><a href="https://travis-ci.org/rpreen/xcsf"><img src="http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=linux&label=build&branch=master"></a></td>
        <td><a href="https://travis-ci.org/rpreen/xcsf"><img src="http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=osx&label=build&branch=master"></a></td>
        <td><a href="https://ci.appveyor.com/project/rpreen/xcsf"><img src="https://ci.appveyor.com/api/projects/status/s4xge68jmlbam005?svg=true"></a></td>
    </tr>
</table>

<table>
    <tr>
        <th>Codacy</th>
        <th>LGTM</th>
        <th>CodeFactor</th>
        <th>Code Inspector</th>
        <th rowspan=2><a href="https://sonarcloud.io/dashboard?id=rpreen_xcsf"><img src="https://sonarcloud.io/api/project_badges/quality_gate?project=rpreen_xcsf"></a></th>
    </tr>
    <tr>
        <td><a href="https://www.codacy.com/app/rpreen/xcsf"><img src="https://api.codacy.com/project/badge/Grade/2213b9ad4e034482bf058d4598d1618b"></a></td>
        <td><a href="https://lgtm.com/projects/g/rpreen/xcsf/context:cpp"><img src="https://img.shields.io/lgtm/grade/cpp/g/rpreen/xcsf.svg?logo=lgtm&logoWidth=18"></a></td>
        <td><a href="https://www.codefactor.io/repository/github/rpreen/xcsf"><img src="https://www.codefactor.io/repository/github/rpreen/xcsf/badge"></a></td>
        <td><a href="https://www.code-inspector.com/public/project/2064/xcsf/dashboard"><img src="https://www.code-inspector.com/project/2064/status/svg"></a></td>
    </tr>
</table>

*******************************************************************************

## Features

Implements both [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) via the updating of match set errors directly and [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) via the updating of action set predictions with an environment reward.

See `default.ini` for a full list of options.

### Evolved Conditions

* `COND_TYPE = 0` : Always matching dummy condition
* `COND_TYPE = 1` : Hyperrectangles
* `COND_TYPE = 2` : Hyperellipsoids
* `COND_TYPE = 3` : Multilayer perceptron neural networks
* `COND_TYPE = 4` : GP trees
* `COND_TYPE = 5` : Dynamical GP graphs
* `COND_TYPE = 6` : Ternary bitstrings (binarises inputs)
* `COND_TYPE = 11` : Both conditions and actions in single dynamical GP graphs
* `COND_TYPE = 12` : Both conditions and actions in single neural networks

### Evolved Actions

* `ACT_TYPE = 0` : Integer actions

### Computed Predictions

* `PRED_TYPE = 0` : Piece-wise constant
* `PRED_TYPE = 1` : Linear least squares
* `PRED_TYPE = 2` : Quadratic least squares
* `PRED_TYPE = 3` : Linear recursive least squares
* `PRED_TYPE = 4` : Quadratic recursive least squares
* `PRED_TYPE = 5` : Stochastic gradient descent multilayer perceptron neural networks

*******************************************************************************

## Compiler options

* `XCSF_PYLIB = ON` : Python library (CMake default = OFF)
* `PARALLEL = ON` : CPU parallelised matching, predicting, and updating with OpenMP (CMake default = ON)
* `ENABLE_TESTS = ON` : Build and execute unit tests (CMake default = OFF)
  
*******************************************************************************

## Requirements

### Stand-alone binary
 
* C11 compliant compiler.
* [CMake](https://www.cmake.org "CMake") (>= 3.12)
* [OpenMP](https://www.openmp.org "OpenMP") (Optional: PARALLEL=ON): supported by [GCC](https://gcc.gnu.org "GCC") and [Clang](https://clang.llvm.org "clang") with libomp.

### Python library
 
* All of the above for building the stand-alone executable.
* C++11 compliant compiler.
* [Python](https://www.python.org "Python") (>= 3)

*******************************************************************************

## Building

### Ubuntu

18.04 / 20.04

```
$ sudo apt install python3 python3-dev cmake
$ git clone --recurse-submodules https://github.com/rpreen/xcsf.git
$ cd xcsf/build
$ cmake -DCMAKE_BUILD_TYPE=Release -DXCSF_PYLIB=ON -DENABLE_TESTS=ON ..
$ make
```

### OSX

XCode 10.1 + Clang

```
$ brew install libomp cmake python
$ git clone --recurse-submodules https://github.com/rpreen/xcsf.git
$ cd xcsf/build
$ cmake -DCMAKE_BUILD_TYPE=Release -DXCSF_PYLIB=ON -DENABLE_TESTS=ON ..
$ make
```

### Windows

[MinGW64-gcc-8.1.0](http://mingw-w64.org) + [Python 3.6.6 x86-64](https://python.org/downloads/windows/)

```
$ git clone --recurse-submodules https://github.com/rpreen/xcsf.git
$ cd xcsf\build
$ cmake -DCMAKE_BUILD_TYPE=Release -DXCSF_PYLIB=ON -DENABLE_TESTS=ON -G "MinGW Makefiles" ..
$ cmake --build . --config Release
```

### Documentation

[Doxygen](http://www.doxygen.nl/download.html) + [graphviz](https://www.graphviz.org/download/)

After running cmake:

```
$ make doc
```

Alternatively see: [XCSF documentation](https://rpreen.github.io/xcsf/).

*******************************************************************************

## Running

### Stand-alone

There are currently 3 built-in problem environments: {csv, mp, maze}.

Example real-multiplexer classification:

```
$ ./xcsf/main mp 6
```

Example discrete mazes:

```
$ ./xcsf/main maze ../env/maze/maze4.txt
```

Example regression: learning `env/csv/sine_3var_train.csv` and testing `env/csv/sine_3var_test.csv`

```
$ ./xcsf/main csv ../env/csv/sine_3var
```

### Python library

After building with CMake option: `-DXCSF_PYLIB=ON`


Single-step reinforcement learning example:

```
$ python example_rmux.py
```

Multi-step reinforcement learning example:

```
$ python example_maze.py
```

Supervised regression learning example:

```
$ python example_regression.py
```

Supervised classification learning example:

```
$ python example_classification.py
```

*******************************************************************************

## Contributing

Contributions are always welcome.

*******************************************************************************

## License

This project is released under the GNU Public License v3.
