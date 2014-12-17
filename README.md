XCSF learning classifier system for function approximation.

The parameter XCSF_EXPONENT sets the number of polynomial exponents used for
computing the prediction; e.g., 1 = linear, 2 = quadratic, 3 = cubic, etc.
Updates are performed with the modified delta update (also known as the
Widrow-Hoff or least mean squares update). Quadratic Recursive Least Squares
(RLS) computed prediction can be enabled by compiling with RLS=1 (in this case
XCSF_EXPONENT is ignored.)

Conditions are represented as either real-valued intervals as in the original
XCSF, or as MLP neural networks if compiled with NEURAL=1.  

Self-adaptive mutation rates can be toggled by compiling with SAM=1 or SAM=0.

An updated GNUPlot of the current system error can be enabled by compiling with
GNUPLOT=1 (on GNU/Linux gnuplot-x11 must be installed.)
