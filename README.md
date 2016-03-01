===========================================================
XCSF learning classifier system for function approximation.
===========================================================
Evolved Conditions:<br>
CON = 0: real-valued hyperrectangle intervals<br>
CON = 1: multilayer perceptron neural networks<br>
CON = 2: GP trees<br>

Mutation for conditions:<br>
SAM = 0: fixed rate<br>
SAM = 1: self-adaptive rate<br>

Computed Predictions:<br>
PRE = 0: linear least squares (aka modified Delta update or Widrow-Hoff).<br>
PRE = 1: quadratic least squares<br>
PRE = 2: linear recursive least squares<br>
PRE = 3: quadratic recursive least squares<br>
PRE = 4: backpropagation multilayer perceptron neural networks<br>

An updated GNUPlot of the current system error can be enabled by compiling with<br>
GNUPLOT=1 (on GNU/Linux gnuplot-x11 must be installed.)<br>

The matching and set prediction functions (where most processing occurs) can be<br>
parallelised using OpenMP by compiling with PARALLEL=1.<br>

--------------
Example usage:
--------------
To initiate the learning of in/sine_1var_train.dat and test performance on
in/sine_1var_test.dat run: 

xcsf sine_1var
