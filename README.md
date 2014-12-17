XCSF learning classifier system for function approximation.

Linear or quadratic (if compiled with QUADRATIC=1) computed prediction, updated
with the modified Delta update (also known as the Widrow-Hoff or least mean
squares update.) Recursive least squares (RLS) update can be enabled by
compiling with RLS=1.

Conditions are represented as either real-valued intervals as in the original
XCSF, or as MLP neural networks if compiled with NEURAL=1.  

Self-adaptive mutation rates can be toggled by compiling with SAM=1 or SAM=0.

An updated GNUPlot of the current system error can be enabled by compiling with
GNUPLOT=1 (on GNU/Linux gnuplot-x11 must be installed.)
