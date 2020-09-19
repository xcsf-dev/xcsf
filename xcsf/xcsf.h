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

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int VERSION_MAJOR = 1; //!< XCSF major version number
static const int VERSION_MINOR = 0; //!< XCSF minor version number
static const int VERSION_BUILD = 0; //!< XCSF build version number

#define COND_TYPE_DUMMY (0) //!< Condition type dummy
#define COND_TYPE_HYPERRECTANGLE (1) //!< Condition type hyperrectangle
#define COND_TYPE_HYPERELLIPSOID (2) //!< Condition type hyperellipsoid
#define COND_TYPE_NEURAL (3) //!< Condition type neural network
#define COND_TYPE_GP (4) //!< Condition type tree GP
#define COND_TYPE_DGP (5) //!< Condition type DGP
#define COND_TYPE_TERNARY (6) //!< Condition type ternary
#define RULE_TYPE_DGP (11) //!< Condition type and action type DGP
#define RULE_TYPE_NEURAL (12) //!< Condition type and action type neural
#define RULE_TYPE_NETWORK (13) //!< Condition type and prediction type neural

#define ACT_TYPE_INTEGER (0) //!< Action type integer

#define PRED_TYPE_CONSTANT (0) //!< Prediction type constant
#define PRED_TYPE_NLMS_LINEAR (1) //!< Prediction type linear nlms
#define PRED_TYPE_NLMS_QUADRATIC (2) //!< Prediction type quadratic nlms
#define PRED_TYPE_RLS_LINEAR (3) //!< Prediction type linear rls
#define PRED_TYPE_RLS_QUADRATIC (4) //!< Prediction type quadratic rls
#define PRED_TYPE_NEURAL (5) //!< Prediction type neural

#define MAX_LAYERS (100) //!< Maximum number of neural network layers

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
    struct CL *cl; //!< Pointer to classifier data structure
    struct CLIST *next; //!< Pointer to the next list element
} CLIST;

/**
 * @brief Classifier set.
 */
typedef struct SET {
    struct CLIST *list; //!< Linked list of classifiers
    int size; //!< Number of macro-classifiers
    int num; //!< The total numerosity of classifiers
} SET;

/**
 * @brief XCSF data structure.
 */
typedef struct XCSF {
    double error; //!< Average system error
    double msetsize; //!< Average match set size
    double asetsize; //!< Average action set size
    double mfrac; //!< Generalisation measure
    double GAMMA; //!< Discount factor for multi-step reward
    double P_EXPLORE; //!< Probability of exploring vs. exploiting
    double ALPHA; //!< Linear coefficient used to calculate classifier accuracy
    double BETA; //!< Learning rate for updating error, fitness, and set size
    double DELTA; //!< Fraction of population to increase deletion vote
    double EPS_0; //!< Target error under which classifier accuracy is set to 1
    double ERR_REDUC; //!< Amount to reduce an offspring's error
    double FIT_REDUC; //!< Amount to reduce an offspring's fitness
    double INIT_ERROR; //!< Initial classifier error value
    double INIT_FITNESS; //!< Initial classifier fitness value
    double NU; //!< Exponent used in calculating classifier accuracy
    double P_CROSSOVER; //!< Probability of applying crossover
    double THETA_EA; //!< Average match set time between EA invocations
    double EA_SELECT_SIZE; //!< Fraction of set size for tournaments
    double COND_MAX; //!< Maximum value expected from inputs
    double COND_MIN; //!< Minimum value expected from inputs
    double COND_SMIN; //!< Minimum initial spread for hyperectangles etc.
    double *gp_cons; //!< Stores constants available for GP trees
    double COND_ETA; //!< Gradient descent rate for updating the condition
    double PRED_ETA; //!< Gradient desecnt rate for updating the prediction
    double PRED_X0; //!< Prediction weight vector offset value
    double PRED_RLS_SCALE_FACTOR; //!< Initial values for the RLS gain-matrix
    double PRED_RLS_LAMBDA; //!< Forget rate for RLS
    double PRED_MOMENTUM; //!< Momentum for gradient descent
    double PRED_DECAY; //!< Weight decay for gradient descent
    struct EnvVtbl const *env_vptr; //!< Functions acting on environments
    void *env; //!< Environment structure
    double *pa; //!< Prediction array (stores fitness weighted predictions)
    double *nr; //!< Prediction array (stores total fitness)
    double prev_reward; //!< Reward from previous step in a multi-step trial
    double prev_pred; //!< Payoff prediction made on the previous step
    double *prev_state; //!< Environment state on the previous step
    double (*loss_ptr)(const struct XCSF *, const double *,
                       const double *); //!< Error function
    SET pset; //!< Population set
    SET prev_pset; //!< Previously stored population set
    SET mset; //!< Match set
    SET aset; //!< Action set
    SET kset; //!< Kill set
    SET prev_aset; //!< Previous action set
    int time; //!< Current number of EA executions
    int OMP_NUM_THREADS; //!< Number of threads for parallel processing
    int MAX_TRIALS; //!< Number of problem instances to run in one experiment
    int PERF_TRIALS; //!< Number of problem instances to avg performance output
    int POP_SIZE; //!< Maximum number of micro-classifiers in the population
    int LOSS_FUNC; //!< Which loss/error function to apply
    int TELETRANSPORTATION; //!< Maximum steps for a multi-step problem
    int THETA_DEL; //!< Min experience before fitness used during deletion
    int COND_TYPE; //!< Classifier condition type: hyperrectangles, etc.
    int PRED_TYPE; //!< Classifier prediction type: least squares, etc.
    int ACT_TYPE; //!< Classifier action type
    int M_PROBATION; //!< Trials since creation a cl must match at least 1 input
    int LAMBDA; //!< Number of offspring to create each EA invocation
    int EA_SELECT_TYPE; //!< Roulette or tournament for EA parental selection
    int COND_BITS; //!< Bits per float to binarise inputs for ternary conditions
    int MAX_K; //!< Maximum number of connections a DGP node may have
    int MAX_T; //!< Maximum number of cycles to update a DGP graph
    int GP_NUM_CONS; //!< Number of constants available for GP trees
    int GP_INIT_DEPTH; //!< Initial depth of GP trees
    int MAX_NEURON_GROW; //!< Max num of neurons to add/remove during mutation
    int COND_OUTPUT_ACTIVATION; //!< Condition output activation function
    int COND_HIDDEN_ACTIVATION; //!< Condition hidden activation function
    int PRED_OUTPUT_ACTIVATION; //!< Prediction output activation function
    int PRED_HIDDEN_ACTIVATION; //!< Prediction hidden activation function
    int THETA_SUB; //!< Minimum experience of a classifier to become a subsumer
    int pa_size; //!< Prediction array size
    int x_dim; //!< Number of problem input variables
    int y_dim; //!< Number of problem output variables
    int n_actions; //!< Number of class labels / actions
    int COND_NUM_NEURONS[MAX_LAYERS]; //!< Condition initial number of neurons
    int COND_MAX_NEURONS[MAX_LAYERS]; //!< Condition maximum number of neurons
    int PRED_NUM_NEURONS[MAX_LAYERS]; //!< Prediction initial number of neurons
    int PRED_MAX_NEURONS[MAX_LAYERS]; //!< Prediction maximum number of neurons
    _Bool POP_INIT; //!< Pop initially empty or filled with random conditions
    _Bool STATEFUL; //!< Whether classifiers should retain state across trials
    _Bool COND_EVOLVE_WEIGHTS; //!< Evolve condition weights
    _Bool COND_EVOLVE_NEURONS; //!< Evolve number of condition neurons
    _Bool COND_EVOLVE_FUNCTIONS; //!< Evolve condition activations
    _Bool COND_EVOLVE_CONNECTIVITY; //!< Evolve condition connections
    _Bool PRED_RESET; //!< Reset offspring predictions instead of copying
    _Bool PRED_EVOLVE_WEIGHTS; //!< Evolve prediction weights
    _Bool PRED_EVOLVE_NEURONS; //!< Evolve number of prediction neurons
    _Bool PRED_EVOLVE_FUNCTIONS; //!< Evolve prediction activations
    _Bool PRED_EVOLVE_CONNECTIVITY; //!< Evolve prediction connections
    _Bool PRED_EVOLVE_ETA; //!< Evolve prediction gradient descent rates
    _Bool PRED_SGD_WEIGHTS; //!< Whether to use gradient descent for predictions
    _Bool EA_SUBSUMPTION; //!< Whether to try and subsume offspring classifiers
    _Bool SET_SUBSUMPTION; //!< Whether to perform match set subsumption
    _Bool explore; //!< Whether the system is currently exploring or exploiting
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

size_t
xcsf_load(struct XCSF *xcsf, const char *filename);

size_t
xcsf_save(const struct XCSF *xcsf, const char *filename);

void
xcsf_free(struct XCSF *xcsf);

void
xcsf_init(struct XCSF *xcsf);

void
xcsf_print_pop(const struct XCSF *xcsf, const _Bool print_cond,
               const _Bool print_act, const _Bool print_pred);

void
xcsf_ae_to_classifier(struct XCSF *xcsf, const int y_dim, const int n_del);

void
xcsf_pred_expand(const struct XCSF *xcsf);

void
xcsf_retrieve_pop(XCSF *xcsf);

void
xcsf_store_pop(XCSF *xcsf);
