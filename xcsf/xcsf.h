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
   
/**
 * @file xcsf.h
 * @brief XCSF data structures.
 */ 
 
#pragma once

/**
 * @brief Classifier data structure.
 */
typedef struct CL {
    struct CondVtbl const *cond_vptr; //!< functions acting on conditions
    struct PredVtbl const *pred_vptr; //!< functions acting on predictions
    struct ActVtbl const *act_vptr; //!< functions acting on actions
    void *cond; //!< condition structure
    void *pred; //!< prediction structure
    void *act; //!< action structure
    double *mu; //!< self-adaptive mutation rates
    double err; //!< error
    double fit; //!< fitness
    int num; //!< numerosity
    int exp; //!< experience
    double size; //!< average participated set size
    int time; //!< time EA last executed in a participating set
    _Bool m; //!< whether the classifier matches current input
    double *prediction; //!< current classifier prediction
    int action; //!< current classifier action
    _Bool *mhist; //!< (theta_sub) recent matching decisions
} CL;

/**
 * @brief Classifier linked list
 */
typedef struct CLIST {
    CL *cl; //!< pointer to classifier data structure
    struct CLIST *next; //!< pointer to the next list element
} CLIST;

/**
 * @brief Classifier set
 */
typedef struct SET {
    CLIST *list; //!< linked list of classifiers
    int size; //!< number of macro-classifiers
    int num; //!< the total numerosity of classifiers
} SET;

/**
 * @brief XCSF data structure
 */
typedef struct XCSF {
    SET pset; //!< population set
    int time; //!< current number of executed trials
    double msetsize; //!< average match set size

    // experiment parameters
    int OMP_NUM_THREADS; //!< number of threads for parallel processing
    _Bool POP_INIT; //!< population initially empty or filled with random conditions
    int THETA_MNA; //!< minimum number of classifiers in a match set
    int MAX_TRIALS; //!< number of problem instances to run in one experiment
    int PERF_AVG_TRIALS; //!< number of problem instances to average performance output
    int POP_SIZE; //!< maximum number of macro-classifiers in the population
    int LOSS_FUNC; //!< which loss/error function to apply

    // multi-step problem parameters
    double GAMMA; //!< discount factor in calculating the reward for multi-step problems
    int TELETRANSPORTATION; //!< num steps to reset a multi-step problem if goal not found

    // classifier parameters
    double ALPHA; //!< linear coefficient used in calculating classifier accuracy
    double BETA; //!< learning rate for updating error, fitness, and set size
    double DELTA; //!< fit used in prob of deletion if fit less than this frac of avg pop fit 
    double EPS_0; //!< classifier target error, under which the fitness is set to 1
    double ERR_REDUC; //!< amount to reduce an offspring's error
    double FIT_REDUC; //!< amount to reduce an offspring's fitness
    double INIT_ERROR; //!< initial classifier error value
    double INIT_FITNESS; //!< initial classifier fitness value
    double NU; //!< exponent used in calculating classifier accuracy
    int THETA_DEL; //!< min experience before fitness used in probability of deletion
    int COND_TYPE; //!< classifier condition type: hyperrectangles, GP trees, etc.
    int PRED_TYPE; //!< classifier prediction type: least squares, neural nets, etc.
    int ACT_TYPE; //!< classifier action type

    // evolutionary algorithm parameters
    double P_CROSSOVER; //!< probability of applying crossover (for hyperrectangles)
    double P_MUTATION; //!< probability of mutation occuring per allele
    double F_MUTATION; //!< probability of performing mutating a graph/net function
    double S_MUTATION; //!< maximum amount to mutate an allele
    double E_MUTATION; //!< rate of gradient descent mutation
    double THETA_EA; //!< average match set time between EA invocations
    int LAMBDA; //!< number of offspring to create each EA invocation
    int EA_SELECT_TYPE; //!< roulette or tournament for EA parental selection
    double EA_SELECT_SIZE; //!< fraction of set size for tournaments

    // self-adaptive mutation parameters
    int SAM_TYPE; //!< 0 = log normal, 1 = ten normally distributed rates
    int SAM_NUM; //!< number of self-adaptive mutation rates
    double SAM_MIN; //!< minimum value of a log normal adaptive mutation rate

    // classifier condition parameters
    double MAX_CON; //!< maximum value expected from inputs
    double MIN_CON; //!< minimum value expected from inputs
    int DGP_NUM_NODES; //!< number of nodes in a DGP graph
    _Bool RESET_STATES; //!< whether to reset the initial states of DGP graphs
    int MAX_K; //!< maximum number of connections a DGP node may have
    int MAX_T; //!< maximum number of cycles to update a DGP graph
    int GP_NUM_CONS; //!< number of constants available for GP trees
    int GP_INIT_DEPTH; //!< initial depth of GP trees
    double *gp_cons; //!< stores constants available for GP trees

    double COND_ETA; //!< gradient descent rate for updating the condition
    _Bool COND_EVOLVE_WEIGHTS; //!< whether to evolve condition network weights
    _Bool COND_EVOLVE_NEURONS; //!< whether to evolve number of condition network neurons
    _Bool COND_EVOLVE_FUNCTIONS; //!< whether to evolve condition network activation functions
    int COND_NUM_HIDDEN_NEURONS; //!< initial number of hidden neurons (random if <= 0)
    int COND_MAX_HIDDEN_NEURONS; //!< maximum number of neurons if evolved
    int COND_HIDDEN_NEURON_ACTIVATION; //!< activation function for the hidden layer
 
    // prediction parameters
    double PRED_ETA; //!< gradient desecnt rate for updating the prediction
    double PRED_X0; //!< prediction weight vector offset value
    double PRED_RLS_SCALE_FACTOR; //!< initial diagonal values of the RLS gain-matrix
    double PRED_RLS_LAMBDA; //!< forget rate for RLS: small values may be unstable
    _Bool PRED_EVOLVE_WEIGHTS; //!< whether to evolve prediction network weights
    _Bool PRED_EVOLVE_NEURONS; //!< whether to evolve number of prediction network neurons
    _Bool PRED_EVOLVE_FUNCTIONS; //!< whether to evolve prediction network activation functions
    _Bool PRED_EVOLVE_ETA; //!< whether to evolve prediction gradient descent rates
    _Bool PRED_SGD_WEIGHTS; //!< whether to use gradient descent for predictions
    double PRED_MOMENTUM; //!< momentum for gradient descent
    int PRED_NUM_HIDDEN_NEURONS; //!< initial number of hidden neurons (random if <= 0)
    int PRED_MAX_HIDDEN_NEURONS; //!< maximum number of neurons if evolved
    int PRED_HIDDEN_NEURON_ACTIVATION; //!< activation function for the hidden layer

    // subsumption parameters
    _Bool EA_SUBSUMPTION; //!< whether to try and subsume offspring classifiers
    _Bool SET_SUBSUMPTION; //!< whether to perform match set subsumption
    int THETA_SUB; //!< minimum experience of a classifier to become a subsumer

    // built-in environments
    struct EnvVtbl const *env_vptr; //!< functions acting on environments
    void *env; // !< environment structure

    // set by environment
    int stage; //!< current stage of training
    _Bool train; //!< training or test mode
    int num_x_vars; //!< number of problem input variables
    int num_y_vars; //!< number of problem output variables
    int num_actions; //!< number of class labels / actions
    double (*loss_ptr)(struct XCSF*, double*, double*); //!< pointer to loss/error function
} XCSF;                  

/**
 * @brief Input data structure
 */
typedef struct INPUT {
    double *x; //!< feature variables
    double *y; //!< target variables
    int x_cols; //!< number of feature variables
    int y_cols; //!< number of target variables
    int rows; //!< number of instances
} INPUT;

double xcsf_fit1(XCSF *xcsf, INPUT *train_data, _Bool shuffle);
double xcsf_fit2(XCSF *xcsf, INPUT *train_data, INPUT *test_data, _Bool shuffle);
double xcsf_score(XCSF *xcsf, INPUT *test_data);
void xcsf_predict(XCSF *xcsf, double *input, double *output, int rows);
void xcsf_print_match_set(XCSF *xcsf, double *input, _Bool printc, _Bool printa, _Bool printp);
void xcsf_print_pop(XCSF *xcsf, _Bool printc, _Bool printa, _Bool printp);
size_t xcsf_load(XCSF *xcsf, char *fname);
size_t xcsf_save(XCSF *xcsf, char *fname);
double xcsf_version();
