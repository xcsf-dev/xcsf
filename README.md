# XCSF learning classifier system

An implementation of the XCSF [learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) that can be built as a stand-alone binary or as a Python library. XCSF is an accuracy-based [online](https://en.wikipedia.org/wiki/Online_machine_learning) [evolutionary](https://en.wikipedia.org/wiki/Evolutionary_computation) [machine learning](https://en.wikipedia.org/wiki/Machine_learning) system with locally approximating functions that compute classifier payoff prediction directly from the input state. It can be seen as a generalisation of XCS where the prediction is a scalar value. XCSF attempts to find solutions that are accurate and maximally general over the global input space, similar to most machine learning techniques. However, it maintains the additional power to subdivide the input space into simpler local approximations.

*******************************************************************************

<table>
    <tr>
        <th>License</th>
        <th>Linux Build</th>
        <th>OSX Build</th>
        <th>Windows Build</th>
        <th>Doxygen CI</th>
    </tr>
    <tr>
        <td><a href="http://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg"></a></td>
        <td><a href="https://travis-ci.com/rpreen/xcsf"><img src="https://travis-matrix-badges.herokuapp.com/repos/rpreen/xcsf/branches/master/2?use_travis_com=true"></a></td>
        <td><a href="https://travis-ci.com/rpreen/xcsf"><img src="https://travis-matrix-badges.herokuapp.com/repos/rpreen/xcsf/branches/master/3?use_travis_com=true"></a></td>
        <td><a href="https://ci.appveyor.com/project/rpreen/xcsf"><img src="https://ci.appveyor.com/api/projects/status/s4xge68jmlbam005?svg=true"></a></td>
        <td><a href="https://travis-ci.com/rpreen/xcsf"><img src="https://travis-matrix-badges.herokuapp.com/repos/rpreen/xcsf/branches/master/4?use_travis_com=true"></a></td>
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

## XCSF Overview

XCSF is [rule-based](https://en.wikipedia.org/wiki/Rule-based_machine_learning) and maintains a population of classifiers where each classifier *cl* consists of:

- a condition structure *cl.C* that determines whether the rule matches input ***x***
- an action structure *cl.A* that selects an action *a* to be performed for a given ***x***
- a prediction structure *cl.P* that computes the expected payoff for performing *a* upon receipt of ***x***

In addition, each classifier maintains a measure of its experience *exp*, error &epsilon;, fitness *F*, numerosity *num*, average participated set size *as*, and the time stamp *ts* of the last [evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm) (EA) invocation on a participating set.

For each step within a learning trial, XCSF constructs a match set [M] composed of classifiers in the population set [P] whose *cl.C* matches ***x***. If [M] contains fewer than *&theta;*<sub>mna</sub> actions, a covering mechanism generates classifiers with matching *cl.C* and random *cl.A*.

For each possible action *a<sub>k</sub>* in [M], XCSF estimates the expected payoff by computing the fitness-weighted average prediction *P*(*a<sub>k</sub>*). That is, for each action *k* and classifier prediction *p<sub>j</sub>* in [M], the system prediction *P<sub>k</sub> = &sum;<sub>j</sub> F<sub>j</sub>p<sub>j</sub> / &sum;<sub>j</sub>F<sub>j</sub>*.

A system action is then randomly or probabilistically selected during exploration, and the highest payoff action *P<sub>k</sub>* used during exploitation. Classifiers in [M] advocating the chosen action are subsequently used to construct an action set [A]. The action is then performed and a scalar reward *r* &isin; &real; received, along with the next sensory input.

In a single-step problem, each classifier *cl<sub>j</sub>* &isin; [A] has its experience incremented and fitness, error, and set size updated using the Widrow-Hoff [delta rule](https://en.wikipedia.org/wiki/Delta_rule) with learning rate *&beta;* &isin; [0,1] as follows.

- Error: *&epsilon;<sub>j</sub> &larr; &epsilon;<sub>j</sub> + &beta;* (| *r* &minus; *p<sub>j</sub>* | &minus; *&epsilon;<sub>j</sub>*)
- Accuracy: *&kappa;<sub>j</sub>* =
    * 1 if *&epsilon;<sub>j</sub> < &epsilon;<sub>0</sub>*
    * *&alpha;* ( *&epsilon;<sub>j</sub> / &epsilon;<sub>0</sub>* )<sup>&minus;*&nu;*</sup> otherwise.
<br>With target error threshold *&epsilon;<sub>0</sub>* and accuracy offset *&alpha;* &isin; [0,1], and slope *&nu;*.
- Relative accuracy: *&kappa;<sub>j</sub>'* = (*&kappa;<sub>j</sub> &middot; num<sub>j</sub>*) / *&sum;<sub>j</sub> &kappa;<sub>j</sub> &middot; num<sub>j</sub>*
- Fitness: *F<sub>j</sub> &larr; F<sub>j</sub> + &beta;*(*&kappa;<sub>j</sub>' &minus; F<sub>j</sub>*)
- Set size estimate: *as<sub>j</sub> &larr; as<sub>j</sub> + &beta;*(|[A]| &minus; *as<sub>j</sub>*)

Thereafter, *cl.C*, *cl.A*, and *cl.P* are updated according to the representation adopted.

The EA is applied to classifiers within [A] if the average time since its previous execution exceeds *&theta;*<sub>EA</sub>. Upon invocation, the *ts* of each classifier is updated. Two parents are chosen based on their fitness via [roulette wheel selection](https://en.wikipedia.org/wiki/Fitness_proportionate_selection) (or [tournament](https://en.wikipedia.org/wiki/Tournament_selection)) and *&lambda;* number of offspring are created via [crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) with probability *&chi;* and [mutation](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)) with probability *&mu;*.

Offspring parameters are initialised by setting the error and fitness to the parental average, and discounted by reduction parameters for error *&epsilon;<sub>R</sub>* and fitness *F<sub>R</sub>*. Offspring *exp* and *num* are set to one. If subsumption is enabled and the offspring are subsumed by either parent with sufficient accuracy (*&epsilon;<sub>j</sub>* &lt; *&epsilon;<sub>0</sub>*) and experience (*exp<sub>j</sub> &gt; &theta;*<sub>sub</sub>) it is not included in [P]; instead the parents' *num* is incremented.

The resulting offspring are added to [P] and the maximum (micro-classifier) population size *N* is enforced by removing classifiers selected via roulette with the deletion vote.

The deletion vote is set proportionally to the set size estimate *as*. However, the vote is increased by a factor *F̅ / F<sub>j</sub>* for classifiers that are sufficiently experienced (*exp<sub>j</sub> &gt; &theta;*<sub>del</sub>) and with small fitness (*F<sub>j</sub> &lt; &delta;F̅*) where *F̅* is the [P] mean fitness, and typically *&delta;* = 0.1.

In a multi-step problem, the previous action set [A]<sub>-1</sub> is instead updated using a *&gamma;* &isin; [0,1] discounted reward (similar to [*Q*-learning](https://en.wikipedia.org/wiki/Q-learning)) and the EA may be run therein.

<img src="doc/schematic.svg">

Schematic illustration (shown above) of XCSF for reinforcement learning. For supervised learning, a single (dummy) action is used such that [A] = [M] and the system prediction is made directly accessible to the environment.
                                                                                                 
A number of interacting pressures have been identified. A set pressure provides more frequent reproduction opportunities for more general rules. In opposition is a fitness pressure which represses the reproduction of inaccurate and over-general rules. Many forms of *cl.C*, *cl.A*, and *cl.P* have been used for classifier knowledge representation since the original ternary conditions, integer actions, and scalar predictions. Some of these are implemented here.

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
* `COND_TYPE = 12` : Both conditions and actions in single (recurrent) neural networks

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
 
* [C11](https://en.wikipedia.org/wiki/C11_(C_standard_revision)) compliant compiler.
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
