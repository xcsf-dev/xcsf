/*
 * Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

// classifier data structure
typedef struct CL {
    struct CondVtbl const *cond_vptr; // functions acting on conditions
    struct PredVtbl const *pred_vptr; // functions acting on predictions
    struct ActVtbl const *act_vptr; // functions acting on actions
    void *cond; // condition structure
    void *pred; // prediction structure
    void *act; // action structure
    double *mu; // self-adaptive mutation rates
    double err; // error
    double fit; // fitness
    int num; // numerosity
    int exp; // experience
    double size; // average participated set size
    int time; // time EA last executed in a participating set
    _Bool m; // whether the classifier matches current input
    double *prediction; // current classifier prediction
    double *action; // current classifier action
} CL;

// classifier linked list
typedef struct CLIST {
    CL *cl;
    struct CLIST *next;
} CLIST;

// classifier set
typedef struct SET {
    CLIST *list; // linked list of classifiers
    int size; // number of macro-classifiers
    int num; // the total numerosity of classifiers
} SET;

// xcsf data structure
typedef struct XCSF {
    SET pset; // population set
    int time; // current number of executed trials
    double msetsize; // average match set size

    // experiment parameters
    int OMP_NUM_THREADS; // number of threads for parallel processing
    _Bool POP_INIT; // population initially empty or filled with random conditions
    int THETA_MNA; // minimum number of classifiers in a match set
    int MAX_TRIALS; // number of problem instances to run in one experiment
    int PERF_AVG_TRIALS; // number of problem instances to average performance output
    int POP_SIZE; // maximum number of macro-classifiers in the population
    int LOSS_FUNC; // which loss/error function to apply

    // classifier parameters
    double ALPHA; // linear coefficient used in calculating classifier accuracy
    double BETA; // learning rate for updating error, fitness, and set size
    double DELTA; // fit used in prob of deletion if fit less than this frac of avg pop fit 
    double EPS_0; // classifier target error, under which the fitness is set to 1
    double ERR_REDUC; // amount to reduce an offspring's error
    double FIT_REDUC; // amount to reduce an offspring's fitness
    double INIT_ERROR; // initial classifier error value
    double INIT_FITNESS; // initial classifier fitness value
    double NU; // exponent used in calculating classifier accuracy
    double THETA_DEL; // min experience before fitness used in probability of deletion
    int COND_TYPE; // classifier condition type: hyperrectangles, GP trees, etc.
    int PRED_TYPE; // classifier prediction type: least squares, neural nets, etc.
    int ACT_TYPE; // classifier action type

    // evolutionary algorithm parameters
    double P_CROSSOVER; // probability of applying crossover (for hyperrectangles)
    double P_MUTATION; // probability of mutation occuring per allele
    double F_MUTATION; // probability of performing mutating a graph/net function
    double S_MUTATION; // maximum amount to mutate an allele
    double E_MUTATION; // rate of gradient descent mutation
    double THETA_EA; // average match set time between EA invocations
    int LAMBDA; // number of offspring to create each EA invocation

    // self-adaptive mutation parameters
    int SAM_TYPE; // 0 = log normal, 1 = ten normally distributed rates
    int SAM_NUM; // number of self-adaptive mutation rates
    double SAM_MIN; // minimum value of a log normal adaptive mutation rate

    // classifier condition parameters
    double MAX_CON; // maximum value expected from inputs
    double MIN_CON; // minimum value expected from inputs
    int NUM_HIDDEN_NEURONS; // number of hidden neurons to perform matching condition
    int HIDDEN_NEURON_ACTIVATION; // activation function for the hidden layer
    double MOMENTUM; // momentum for gradient descent
    int DGP_NUM_NODES; // number of nodes in a DGP graph
    _Bool RESET_STATES; // whether to reset the initial states of DGP graphs
    int MAX_K; // maximum number of connections a DGP node may have
    int MAX_T; // maximum number of cycles to update a DGP graph
    int MAX_FORWARD;
    int GP_NUM_CONS; // number of constants available for GP trees
    int GP_INIT_DEPTH; // initial depth of GP trees
    double *gp_cons; // stores constants available for GP trees

    // prediction parameters
    double ETA; // learning rate for updating the computed prediction
    double X0; // prediction weight vector offset value
    double RLS_SCALE_FACTOR; // initial diagonal values of the RLS gain-matrix
    double RLS_LAMBDA; // forget rate for RLS: small values may be unstable

    // neural classifier parameters
    _Bool COND_EVOLVE_WEIGHTS;
    _Bool COND_EVOLVE_NEURONS;
    _Bool COND_EVOLVE_FUNCTIONS;
    _Bool PRED_EVOLVE_WEIGHTS;
    _Bool PRED_EVOLVE_NEURONS;
    _Bool PRED_EVOLVE_FUNCTIONS;
    _Bool PRED_EVOLVE_ETA;
    _Bool PRED_SGD_WEIGHTS;

    // subsumption parameters
    _Bool EA_SUBSUMPTION; // whether to try and subsume offspring classifiers
    _Bool SET_SUBSUMPTION; // whether to perform match set subsumption
    double THETA_SUB; // minimum experience of a classifier to become a subsumer

    // set by environment
    int stage; // current stage of training
    _Bool train; // training or test mode
    int num_x_vars; // number of problem input variables
    int num_y_vars; // number of problem output variables
    int num_classes; // number of class labels
    double (*loss_ptr)(struct XCSF*, double*, double*); // pointer to loss/error function
} XCSF;                  

// input data structure
typedef struct INPUT {
    double *x;
    double *y;
    int x_cols;
    int y_cols;
    int rows;
} INPUT;
                  
double xcsf_fit1(XCSF *xcsf, INPUT *train_data, _Bool shuffle);
double xcsf_fit2(XCSF *xcsf, INPUT *train_data, INPUT *test_data, _Bool shuffle);
void xcsf_predict(XCSF *xcsf, double *input, double *output, int rows);
void xcsf_print_match_set(XCSF *xcsf, double *input, _Bool printc, _Bool printa, _Bool printp);
void xcsf_print_pop(XCSF *xcsf, _Bool printc, _Bool printa, _Bool printp);
size_t xcsf_load(XCSF *xcsf, char *fname);
size_t xcsf_save(XCSF *xcsf, char *fname);
double xcsf_version();
