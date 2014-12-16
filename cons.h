/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

void constants_init(int argc, char **argv);

// values are read from cons.txt
int POP_SIZE;
_Bool POP_INIT;
int NUM_EXPERIMENTS;
int MAX_TRIALS;
double P_CROSSOVER;
double P_MUTATION;
double THETA_SUB;
double EPS_0;
double DELTA;
double THETA_DEL;
double THETA_GA;
double THETA_MNA;
int THETA_OFFSPRING;
double BETA;
double ALPHA; 
double NU;
double INIT_FITNESS;
double INIT_ERROR;
double ERR_REDUC;
double FIT_REDUC;
_Bool GA_SUBSUMPTION;
_Bool SET_SUBSUMPTION;
int PERF_AVG_TRIALS;
double XCSF_X0;
double XCSF_ETA;
int XCSF_EXPONENT; // 1 = linear prediction, 2 = quadratic, etc.
int NUM_HIDDEN_NEURONS; // number of hidden neurons to perform matching condition
// self-adaptive mutation
double muEPS_0;
int NUM_MU;
// set by environment
int state_length;
// classifier condition parameters
// depends on the problem function
double S_MUTATION; // max mutation step size
// interval predicate parameters
double MIN_CON;
double MAX_CON;
