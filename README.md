===========================================================
XCSF learning classifier system for function approximation.
===========================================================
Linear or quadratic (if compiled with QUADRATIC=1) computed prediction, updated
with the modified Delta update (also known as the Widrow-Hoff or least mean
squares update.) Recursive least squares update can be enabled by compiling
with PRED=1. MLP neural networks can be used to compute the prediction and
updated with the backpropagation algorithm by compiling with PRED=2.

Conditions are represented as either real-valued intervals as in the original
XCSF, or as MLP neural networks if compiled with COND=1.  

Self-adaptive mutation rates can be enabled by compiling with SAM=1.

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
Martin Butz's XCSJava:
http://illigal.org/wp-content/uploads/illigal/pub/src/XCSJava/

Patrick Stalph and Martin Butz's JavaXCSF:
http://www.wsi.uni-tuebingen.de/lehrstuehle/cognitive-modeling/code/overview.html

Pier Luca Lanzi and Daniele Loiacono's xcslib (C++):
http://sourceforge.net/projects/xcslib/

Jaume Bacardit, Natalio Krasnogor and Maria Franco's GAssist (C++):
http://ico2s.org/software/gassist.html

Ryan Urbanowicz's UCS, XCS, ExSTraCS (python):
http://www.ryanurbanowicz.com/software
