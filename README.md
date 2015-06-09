===========================================================
XCSF learning classifier system for function approximation.
===========================================================
Evolved Conditions:
CON = 0: real-valued hyperrectangle intervals
CON = 1: multilayer perceptron neural networks

Mutation for conditions:
SAM = 0: fixed rate
SAM = 1: self-adaptive rate

Computed Predictions:
PRE = 0: linear least squares (aka modified Delta update or Widrow-Hoff).
PRE = 1: quadratic least squares
PRE = 2: linear recursive least squares
PRE = 3: quadratic recursive least squares
PRE = 4: backpropagation multilayer perceptron neural networks

An updated GNUPlot of the current system error can be enabled by compiling with
GNUPLOT=1 (on GNU/Linux gnuplot-x11 must be installed.)

The matching and set prediction functions (where most processing occurs) can be
parallelised using OpenMP by compiling with PARALLEL=1.

--------------
Example usage:
--------------
To initiate the learning of in/sine_1var_train.dat and test performance on
in/sine_1var_test.dat run: 

xcsf sine_1var

------------------------------------
Some additional sources of LCS code:
------------------------------------
Martin Butz's XCS (java):
http://illigal.org/wp-content/uploads/illigal/pub/src/XCSJava/

Patrick Stalph and Martin Butz's XCSF (java):
http://www.wsi.uni-tuebingen.de/lehrstuehle/cognitive-modeling/code/overview.html

Pier Luca Lanzi and Daniele Loiacono's xcslib (C++):
http://sourceforge.net/projects/xcslib/

Jaume Bacardit, Natalio Krasnogor and Maria Franco's GAssist (C++):
http://ico2s.org/software/gassist.html

Ryan Urbanowicz's UCS, XCS, ExSTraCS (python):
http://www.ryanurbanowicz.com/software
