XCSF learning classifier system
-----------
Evolved Conditions:<br>

COND_TYPE =-1: always matching dummy condition<br>
COND_TYPE = 0: real-valued hyperrectangle intervals<br>
COND_TYPE = 1: multilayer perceptron neural networks<br>
COND_TYPE = 2: GP trees<br>
COND_TYPE = 3: dynamical GP graphs<br>
<br>
COND_TYPE = 11: both conditions and predictions in single dynamical GP graphs<br>
COND_TYPE = 12: both conditions and predictions in single neural networks<br>

Computed Predictions:<br>

PRED_TYPE = 0: linear least squares<br>
PRED_TYPE = 1: quadratic least squares<br>
PRED_TYPE = 2: linear recursive least squares<br>
PRED_TYPE = 3: quadratic recursive least squares<br>
PRED_TYPE = 4: backpropagation multilayer perceptron neural networks<br>

Compiler options:
-----------

Mutation for conditions:<br>
SAM = OFF: fixed rate<br>
SAM = ON: self-adaptive rate<br>

An updated GNUPlot of the current system error can be enabled by compiling with<br>
GNUPLOT = ON (on GNU/Linux gnuplot-x11 must be installed.)<br>

The matching and set prediction functions (where most processing occurs) can be<br>
parallelised using OpenMP by compiling with PARALLEL = ON.<br>

Building:
-----------

0. Clone: `git clone --recurse-submodules git@github.com:rpreen/xcsf.git`
1. Change to the build directory: `cd xcsf/build`
2. Run cmake: `cmake .. -DCMAKE_BUILD_TYPE=RELEASE`
3. Run make: `make`

Running:
-----------

Example learning on `data/sine_1var_train` and testing on `data/sine_1var_test`:

1. Change to the application folder: `cd xcsf`
2. Modify constants as needed: `cons.txt`
3. Run: `./xcsf sine_1var`
