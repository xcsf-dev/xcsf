XCSF learning classifier system
-----------
Evolved Conditions:<br>
CON =-1: always matching dummy condition<br>
CON = 0: real-valued hyperrectangle intervals<br>
CON = 1: multilayer perceptron neural networks<br>
CON = 2: GP trees<br>
CON = 3: dynamical GP graphs<br>
<br>
CON = 11: both conditions and predictions in single dynamical GP graphs<br>
CON = 12: both conditions and predictions in single neural networks<br>

Mutation for conditions:<br>
SAM = OFF: fixed rate<br>
SAM = ON: self-adaptive rate<br>

Computed Predictions:<br>
PRE = 0: linear least squares<br>
PRE = 1: quadratic least squares<br>
PRE = 2: linear recursive least squares<br>
PRE = 3: quadratic recursive least squares<br>
PRE = 4: backpropagation multilayer perceptron neural networks<br>

An updated GNUPlot of the current system error can be enabled by compiling with<br>
GNUPLOT = ON (on GNU/Linux gnuplot-x11 must be installed.)<br>

The matching and set prediction functions (where most processing occurs) can be<br>
parallelised using OpenMP by compiling with PARALLEL = ON.<br>

Building:
-----------

Example with GP tree conditions and neural network predictors:

0. Clone: `git clone --recurse-submodules git@github.com:rpreen/xcsf.git`
1. Change to the build directory: `cd xcsf/build`
2. Run cmake: `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCON=2 -DPRE=4`
3. Run make: `make`

Running:
-----------

Example learning on `data/sine_1var_train` and testing on `data/sine_1var_test`:

1. Change to the application folder: `cd xcsf`
2. Modify constants as needed: `cons.txt`
3. Run: `./xcsf sine_1var`
