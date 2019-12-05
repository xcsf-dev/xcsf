# XCSF learning classifier system

An implementation of the XCSF [learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) that can be built as a stand-alone binary or as a Python library.

*******************************************************************************

<table>
    <tr>
        <th>License</th>
        <th>Linux Build</th>
        <th>OSX Build</th>
        <th>Windows Build</th>
        <th>Fossa</th>
    </tr>
    <tr>
        <td><a href="http://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg"></a></td>
        <td><a href="https://travis-ci.org/rpreen/xcsf"><img src="http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=linux&label=build&branch=master"></a></td>
        <td><a href="https://travis-ci.org/rpreen/xcsf"><img src="http://badges.herokuapp.com/travis/rpreen/xcsf?env=BADGE=osx&label=build&branch=master"></a></td>
        <td><a href="https://ci.appveyor.com/project/rpreen/xcsf"><img src="https://ci.appveyor.com/api/projects/status/s4xge68jmlbam005?svg=true"></a></td>
        <td><a href="https://app.fossa.com/projects/git%2Bgithub.com%2Frpreen%2Fxcsf?ref=badge_shield"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Frpreen%2Fxcsf.svg?type=shield"></a></td>
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

See `default.ini` for a full list of options.

### Evolved Conditions

* `COND_TYPE =-1`: Always matching dummy condition
* `COND_TYPE = 0`: Hyperrectangles
* `COND_TYPE = 1`: Hyperellipsoids
* `COND_TYPE = 2`: Multilayer perceptron neural networks
* `COND_TYPE = 3`: GP trees
* `COND_TYPE = 4`: Dynamical GP graphs
* `COND_TYPE = 11`: Both conditions and actions in single dynamical GP graphs
* `COND_TYPE = 12`: Both conditions and actions in single neural networks

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
$ cmake -DCMAKE_BUILD_TYPE=Release -DXCSF_PYLIB=ON ..
$ make
```
 
### OSX (XCode 10.1 / Clang)

```
$ brew install libomp cmake python boost-python3 gnuplot
$ git clone --recurse-submodules git@github.com:rpreen/xcsf.git
$ cd xcsf/build
$ cmake -DCMAKE_BUILD_TYPE=Release -DXCSF_PYLIB=ON ..
$ make
```

### Windows ([MinGW64-gcc-7.2.0](http://mingw-w64.org "MinGW64-gcc-7.2.0"))

```
$ git clone --recurse-submodules git@github.com:rpreen/xcsf.git
$ cd xcsf/build
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -G "MinGW Makefiles" ..
$ cmake --build . --config Release
```

### Documentation (Doxygen + graphviz)

After running cmake:

```
$ make doc
```

Alternatively see: [XCSF documentation](https://rpreen.github.io/xcsf/ "XCSF documentation").

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

*******************************************************************************

## License

This project is released under the GNU Public License v3.
