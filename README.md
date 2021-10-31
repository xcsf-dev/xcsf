# XCSF learning classifier system

An implementation of the XCSF [learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) that can be built as a stand-alone binary or as a Python library. XCSF is an accuracy-based [online](https://en.wikipedia.org/wiki/Online_machine_learning) [evolutionary](https://en.wikipedia.org/wiki/Evolutionary_computation) [machine learning](https://en.wikipedia.org/wiki/Machine_learning) system with locally approximating functions that compute classifier payoff prediction directly from the input state. It can be seen as a generalisation of XCS where the prediction is a scalar value. XCSF attempts to find solutions that are accurate and maximally general over the global input space, similar to most machine learning techniques. However, it maintains the additional power to adaptively subdivide the input space into simpler local approximations.

See the project [wiki](https://github.com/rpreen/xcsf/wiki) for more details.

*******************************************************************************

[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0)
[![Linux Build](https://img.shields.io/github/workflow/status/rpreen/xcsf/Ubuntu%20build?logo=linux&logoColor=white&style=flat&label=Ubuntu)](https://github.com/rpreen/xcsf/actions?query=workflow%3A%22Ubuntu+build%22)
[![MacOS Build](https://img.shields.io/github/workflow/status/rpreen/xcsf/macOS%20build?logo=apple&logoColor=white&style=flat&label=macOS)](https://github.com/rpreen/xcsf/actions?query=workflow%3A%22macOS+build%22)
[![Windows Build](https://img.shields.io/appveyor/build/rpreen/xcsf?logo=windows&logoColor=white&style=flat&label=Windows)](https://ci.appveyor.com/project/rpreen/xcsf)
[![Latest Version](https://img.shields.io/github/v/release/rpreen/xcsf?style=flat)](https://github.com/rpreen/xcsf/releases)
[![DOI](https://zenodo.org/badge/28035841.svg)](https://zenodo.org/badge/latestdoi/28035841)

[![Codacy](https://img.shields.io/codacy/grade/2213b9ad4e034482bf058d4598d1618b?logo=codacy&style=flat)](https://www.codacy.com/app/rpreen/xcsf)
[![LGTM](https://img.shields.io/lgtm/grade/cpp/g/rpreen/xcsf.svg?logo=LGTM&style=flat)](https://lgtm.com/projects/g/rpreen/xcsf/context:cpp)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/rpreen/xcsf?logo=codefactor&style=flat)](https://www.codefactor.io/repository/github/rpreen/xcsf)
[![Code Inspector](https://www.code-inspector.com/project/2064/status/svg)](https://www.code-inspector.com/public/project/2064/xcsf/dashboard)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=rpreen_xcsf&metric=alert_status)](https://sonarcloud.io/dashboard?id=rpreen_xcsf)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=rpreen_xcsf&metric=ncloc)](https://sonarcloud.io/dashboard?id=rpreen_xcsf)

*******************************************************************************

## Table of Contents

* [Requirements](#requirements)
* [Building](#building)
* [Running](#running)
* [Contributing](#contributing)
* [License](#license)

*******************************************************************************

## Requirements

Stand-alone binary:
 
* [C11](https://en.wikipedia.org/wiki/C11_(C_standard_revision)) compliant compiler.
* [CMake](https://www.cmake.org "CMake") (>= 3.12)
* [OpenMP](https://www.openmp.org "OpenMP") (optional): supported by [GCC](https://gcc.gnu.org "GCC") and [Clang](https://clang.llvm.org "clang") with libomp.

Python library:
 
* All of the above for building the stand-alone executable.
* C++11 compliant compiler.
* [Python](https://www.python.org "Python") (>= 3)

*******************************************************************************

## Building

### Compiler Options

* `XCSF_PYLIB = ON` : Python library (CMake default = OFF)
* `PARALLEL = ON` : CPU parallelised matching, predicting, and updating with OpenMP (CMake default = ON)
* `ENABLE_TESTS = ON` : Build and execute unit tests (CMake default = OFF)
  
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

Notes: use 64-bit versions for both Python and mingw. Tested with: python-3.6.6-amd64.exe and mingw-w64-install.exe (8.1.0; x86_64; posix; seh; 0). Some versions of Python have difficulties compiling. XCSF build should generate only a single warning regarding a redundant redeclaration of 'double round(double)'. This is an issue with Python and mingw having their own rounding functions, but this can safely be ignored. A simple method to get started is: Start -> MinGW-W64 project -> Run terminal -> change to the desired install location and enter the commands below. [Compilation should also be possible within an IDE such as Visual Studio or Eclipse.]

```
$ git clone --recurse-submodules https://github.com/rpreen/xcsf.git
$ cd xcsf\build
$ cmake -DCMAKE_BUILD_TYPE=Release -DXCSF_PYLIB=ON -DENABLE_TESTS=ON -G "MinGW Makefiles" ..
$ cmake --build . --config Release
```

### Documentation

[Doxygen](http://www.doxygen.nl/download.html) + [graphviz](https://www.graphviz.org/download/)

After running CMake:

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

### Python

After building with CMake option: `-DXCSF_PYLIB=ON`

See the Python examples. E.g., single-step reinforcement learning:

```
$ python3 example_rmux.py
```

*******************************************************************************

## Contributing

Contributions are always welcome. See `CONTRIBUTING.md` for details.

*******************************************************************************

## License

This project is released under the GNU Public License v3. See `LICENSE.md` for details.
