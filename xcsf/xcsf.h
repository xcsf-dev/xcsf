/*
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
 */

/**
 * @file xcsf.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief XCSF data structures.
 */

#pragma once

static const int VERSION_MAJOR = 1; //!< XCSF major version number
static const int VERSION_MINOR = 0; //!< XCSF minor version number
static const int VERSION_BUILD = 0; //!< XCSF build version number

#define COND_TYPE_DUMMY 0 //!< Condition type dummy
#define COND_TYPE_HYPERRECTANGLE 1 //!< Condition type hyperrectangle
#define COND_TYPE_HYPERELLIPSOID 2  //!< Condition type hyperellipsoid
#define COND_TYPE_NEURAL 3 //!< Condition type neural network
#define COND_TYPE_GP 4 //!< Condition type tree GP
#define COND_TYPE_DGP 5 //!< Condition type DGP
#define COND_TYPE_TERNARY 6 //!< Condition type ternary
#define RULE_TYPE_DGP 11 //!< Condition type and action type DGP
#define RULE_TYPE_NEURAL 12 //!< Condition type and action type neural

#define ACT_TYPE_INTEGER 0 //!< Action type integer

#define PRED_TYPE_CONSTANT 0 //!< Prediction type constant
#define PRED_TYPE_NLMS_LINEAR 1 //!< Prediction type linear nlms
#define PRED_TYPE_NLMS_QUADRATIC 2 //!< Prediction type quadratic nlms
#define PRED_TYPE_RLS_LINEAR 3 //!< Prediction type linear rls
#define PRED_TYPE_RLS_QUADRATIC 4 //!< Prediction type quadratic rls
#define PRED_TYPE_NEURAL 5 //!< Prediction type neural

#define MAX_LAYERS 100 //!< Maximum number of neural network layers

/**
 * @brief Classifier data structure.
 */
typedef struct CL {
    struct CondVtbl const *cond_vptr; //!< Functions acting on conditions
    struct PredVtbl const *pred_vptr; //!< Functions acting on predictions
    struct ActVtbl const *act_vptr; //!< Functions acting on actions
    void *cond; //!< Condition structure
    void *pred; //!< Prediction structure
    void *act; //!< Action structure
    double err; //!< Error
    double fit; //!< Fitness
    int num; //!< Numerosity
    int exp; //!< Experience
    double size; //!< Average participated set size
    int time; //!< Time EA last executed in a participating set
    _Bool m; //!< Whether the classifier matches current input
    double *prediction; //!< Current classifier prediction
    int action; //!< Current classifier action
    int age; //!< Total number of times match testing been performed
    int mtotal; //!< Total number of times actually matched an input
} CL;

/**
 * @brief Classifier linked list.
 */
typedef struct CLIST {
    CL *cl; //!< Pointer to classifier data structure
    struct CLIST *next; //!< Pointer to the next list element
} CLIST;

/**
 * @brief Classifier set.
 */
typedef struct SET {
    CLIST *list; //!< Linked list of classifiers
    int size; //!< Number of macro-classifiers
    int num; //!< The total numerosity of classifiers
} SET;

/**
 * @brief XCSF data structure.
 */
typedef struct XCSF {
    SET pset; //!< Population set
    SET mset; //!< Match set
    SET aset; //!< Action set
    SET kset; //!< Kill set
    SET prev_aset; //!< Previous action set

    int time; //!< Current number of EA executions
    double msetsize; //!< Average match set size
    double mfrac; //!< Generalisation measure

    // experiment parameters
    int OMP_NUM_THREADS; //!< Number of threads for parallel processing
    _Bool POP_INIT; //!< Population initially empty or filled with random conditions
    int MAX_TRIALS; //!< Number of problem instances to run in one experiment
    int PERF_TRIALS; //!< Number of problem instances to average performance output
    int POP_SIZE; //!< Maximum number of micro-classifiers in the population
    int LOSS_FUNC; //!< Which loss/error function to apply

    // multi-step problem parameters
    double GAMMA; //!< Discount factor in calculating the reward for multi-step problems
    int TELETRANSPORTATION; //!< Num steps to reset a multi-step problem if goal not found
    double P_EXPLORE; //!< Probability of exploring vs. exploiting

    // classifier parameters
    double ALPHA; //!< Linear coefficient used in calculating classifier accuracy
    double BETA; //!< Learning rate for updating error, fitness, and set size
    double DELTA; //!< Fit used in prob of deletion if fit less than this frac of avg pop fit
    double EPS_0; //!< Classifier target error, under which the accuracy is set to 1
    double ERR_REDUC; //!< Amount to reduce an offspring's error
    double FIT_REDUC; //!< Amount to reduce an offspring's fitness
    double INIT_ERROR; //!< Initial classifier error value
    double INIT_FITNESS; //!< Initial classifier fitness value
    double NU; //!< Exponent used in calculating classifier accuracy
    int THETA_DEL; //!< Min experience before fitness used in probability of deletion
    int COND_TYPE; //!< Classifier condition type: hyperrectangles, GP trees, etc.
    int PRED_TYPE; //!< Classifier prediction type: least squares, neural nets, etc.
    int ACT_TYPE; //!< Classifier action type
    int M_PROBATION; //!< Trials since rule creation it must match at least 1 input or be deleted
    int SAM_TYPE; //!< Self-adaptive mutation algorithm

    // evolutionary algorithm parameters
    double P_CROSSOVER; //!< Probability of applying crossover (for hyperrectangles)
    double THETA_EA; //!< Average match set time between EA invocations
    int LAMBDA; //!< Number of offspring to create each EA invocation
    int EA_SELECT_TYPE; //!< Roulette or tournament for EA parental selection
    double EA_SELECT_SIZE; //!< Fraction of set size for tournaments

    // classifier condition parameters
    double COND_MAX; //!< Maximum value expected from inputs
    double COND_MIN; //!< Minimum value expected from inputs
    double COND_SMIN; //!< Minimum initial spread for hyperectangles and hyperellipsoids
    int COND_BITS; //!< Number of bits per float to binarise inputs for ternary conditions
    int DGP_NUM_NODES; //!< Number of nodes in a DGP graph
    _Bool RESET_STATES; //!< Whether to reset the initial states of DGP graphs
    int MAX_K; //!< Maximum number of connections a DGP node may have
    int MAX_T; //!< Maximum number of cycles to update a DGP graph
    int GP_NUM_CONS; //!< Number of constants available for GP trees
    int GP_INIT_DEPTH; //!< Initial depth of GP trees
    double *gp_cons; //!< Stores constants available for GP trees
    int MAX_NEURON_MOD; //!< Maximum number of neurons to add or remove during mutation

    double COND_ETA; //!< Gradient descent rate for updating the condition
    _Bool COND_EVOLVE_WEIGHTS; //!< Whether to evolve condition network weights
    _Bool COND_EVOLVE_NEURONS; //!< Whether to evolve number of condition network neurons
    _Bool COND_EVOLVE_FUNCTIONS; //!< Whether to evolve condition network activation functions
    _Bool COND_EVOLVE_CONNECTIVITY; //!< Whether to evolve condition network connectivity
    int COND_NUM_NEURONS[MAX_LAYERS]; //!< Initial num neurons in each condition hidden layer
    int COND_MAX_NEURONS[MAX_LAYERS]; //!< Max num of neurons in each condition hidden layer
    int COND_OUTPUT_ACTIVATION; //!< Activation function for the condition output layer
    int COND_HIDDEN_ACTIVATION; //!< Activation function for the condition hidden layer

    // prediction parameters
    _Bool PRED_RESET; //!< Whether to reset offspring predictions instead of copying
    double PRED_ETA; //!< Gradient desecnt rate for updating the prediction
    double PRED_X0; //!< Prediction weight vector offset value
    double PRED_RLS_SCALE_FACTOR; //!< Initial diagonal values of the RLS gain-matrix
    double PRED_RLS_LAMBDA; //!< Forget rate for RLS: small values may be unstable
    _Bool PRED_EVOLVE_WEIGHTS; //!< Whether to evolve prediction network weights
    _Bool PRED_EVOLVE_NEURONS; //!< Whether to evolve number of prediction network neurons
    _Bool PRED_EVOLVE_FUNCTIONS; //!< Whether to evolve prediction network activation functions
    _Bool PRED_EVOLVE_CONNECTIVITY; //!< Whether to evolve prediction network connectivity
    _Bool PRED_EVOLVE_ETA; //!< Whether to evolve prediction gradient descent rates
    _Bool PRED_SGD_WEIGHTS; //!< Whether to use gradient descent for predictions
    double PRED_MOMENTUM; //!< Momentum for gradient descent
    int PRED_NUM_NEURONS[MAX_LAYERS]; //!< Initial num neurons in each prediction hidden layer
    int PRED_MAX_NEURONS[MAX_LAYERS]; //!< Max num of neurons in each prediction hidden layer
    int PRED_OUTPUT_ACTIVATION; //!< Activation function for the prediction output layer
    int PRED_HIDDEN_ACTIVATION; //!< Activation function for the prediction hidden layer

    // subsumption parameters
    _Bool EA_SUBSUMPTION; //!< Whether to try and subsume offspring classifiers
    _Bool SET_SUBSUMPTION; //!< Whether to perform match set subsumption
    int THETA_SUB; //!< Minimum experience of a classifier to become a subsumer

    // built-in environments
    struct EnvVtbl const *env_vptr; //!< Functions acting on environments
    void *env; //!< Environment structure

    // prediction array
    int pa_size; //!< Prediction array size
    double *pa; //!< Prediction array (stores fitness weighted predictions)
    double *nr; //!< Prediction array (stores total fitness)

    // multi-step
    double prev_reward; //!< Reward from the previous step in a multi-step trial.
    double prev_pred; //!< Payoff prediction made on the previous step in a multi-step trial.
    double *prev_state; //!< Environment state on the previous step in a multi-step trial.

    // set by environment
    _Bool explore; //!< Whether the system is currently exploring or exploiting
    int x_dim; //!< Number of problem input variables
    int y_dim; //!< Number of problem output variables
    int n_actions; //!< Number of class labels / actions
    double (*loss_ptr)(const struct XCSF *, const double *,
                       const double *); //!< Error function
} XCSF;

/**
 * @brief Input data structure.
 */
typedef struct INPUT {
    double *x; //!< Feature variables
    double *y; //!< Target variables
    int x_dim; //!< Number of feature variables
    int y_dim; //!< Number of target variables
    int n_samples; //!< Number of instances
} INPUT;

size_t xcsf_load(XCSF *xcsf, const char *fname);
size_t xcsf_save(const XCSF *xcsf, const char *fname);
void xcsf_init(XCSF *xcsf);
void xcsf_print_pop(const XCSF *xcsf, _Bool printc, _Bool printa, _Bool printp);
void xcsf_ae_to_classifier(XCSF *xcsf, int y_dim, int n_del);
void xcsf_pred_expand(const XCSF *xcsf);
