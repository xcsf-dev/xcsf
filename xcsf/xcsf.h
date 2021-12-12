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
 * @date 2019--2021.
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
static const int VERSION_MINOR = 1; //!< XCSF minor version number
static const int VERSION_BUILD = 4; //!< XCSF build version number

/**
 * @brief Classifier data structure.
 */
struct Cl {
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
    bool m; //!< Whether the classifier matches current input
    double *prediction; //!< Current classifier prediction
    int action; //!< Current classifier action
    int age; //!< Total number of times match testing been performed
    int mtotal; //!< Total number of times actually matched an input
};

/**
 * @brief Classifier linked list.
 */
struct Clist {
    struct Cl *cl; //!< Pointer to classifier data structure
    struct Clist *next; //!< Pointer to the next list element
};

/**
 * @brief Classifier set.
 */
struct Set {
    struct Clist *list; //!< Linked list of classifiers
    int size; //!< Number of macro-classifiers
    int num; //!< The total numerosity of classifiers
};

/**
 * @brief XCSF data structure.
 */
struct XCSF {
    struct Set pset; //!< Population set
    struct Set prev_pset; //!< Previously stored population set
    struct Set mset; //!< Match set
    struct Set aset; //!< Action set
    struct Set kset; //!< Kill set
    struct Set prev_aset; //!< Previous action set
    struct ArgsAct *act; //!< Action parameters
    struct ArgsCond *cond; //!< Condition parameters
    struct ArgsPred *pred; //!< Prediction parameters
    struct ArgsEA *ea; //!< EA parameters
    struct EnvVtbl const *env_vptr; //!< Functions acting on environments
    void *env; //!< Environment structure (for built-in problems)
    double error; //!< Average system error
    double mset_size; //!< Average match set size
    double aset_size; //!< Average action set size
    double mfrac; //!< Generalisation measure
    double prev_reward; //!< Reward from previous step in a multi-step trial
    double prev_pred; //!< Payoff prediction made on the previous step
    double *pa; //!< Prediction array (stores fitness weighted predictions)
    double *nr; //!< Prediction array (stores total fitness)
    double *prev_state; //!< Environment state on the previous step
    int time; //!< Current number of EA executions
    int pa_size; //!< Prediction array size
    int x_dim; //!< Number of problem input variables
    int y_dim; //!< Number of problem output variables
    int n_actions; //!< Number of class labels / actions
    bool explore; //!< Whether the system is currently exploring or exploiting
    double (*loss_ptr)(const struct XCSF *, const double *,
                       const double *); //!< Error function
    double GAMMA; //!< Discount factor for multi-step reward
    double P_EXPLORE; //!< Probability of exploring vs. exploiting
    double ALPHA; //!< Linear coefficient used to calculate classifier accuracy
    double BETA; //!< Learning rate for updating error, fitness, and set size
    double DELTA; //!< Fraction of population to increase deletion vote
    double E0; //!< Target error under which classifier accuracy is set to 1
    double INIT_ERROR; //!< Initial classifier error value
    double INIT_FITNESS; //!< Initial classifier fitness value
    double NU; //!< Exponent used in calculating classifier accuracy
    double HUBER_DELTA; //!< Delta parameter for Huber loss calculation.
    int OMP_NUM_THREADS; //!< Number of threads for parallel processing
    int MAX_TRIALS; //!< Number of problem instances to run in one experiment
    int PERF_TRIALS; //!< Number of problem instances to avg performance output
    int POP_SIZE; //!< Maximum number of micro-classifiers in the population
    int LOSS_FUNC; //!< Which loss/error function to apply
    int TELETRANSPORTATION; //!< Maximum steps for a multi-step problem
    int THETA_DEL; //!< Min experience before fitness used during deletion
    int M_PROBATION; //!< Trials since creation a cl must match at least 1 input
    int THETA_SUB; //!< Minimum experience of a classifier to become a subsumer
    bool POP_INIT; //!< Pop initially empty or filled with random conditions
    bool SET_SUBSUMPTION; //!< Whether to perform match set subsumption
    bool STATEFUL; //!< Whether classifiers should retain state across trials
    bool COMPACTION; //!< if sys err < E0: largest of 2 roulette spins deleted
};

/**
 * @brief Input data structure.
 */
struct Input {
    double *x; //!< Feature variables
    double *y; //!< Target variables
    int x_dim; //!< Number of feature variables
    int y_dim; //!< Number of target variables
    int n_samples; //!< Number of instances
};

size_t
xcsf_load(struct XCSF *xcsf, const char *filename);

size_t
xcsf_save(const struct XCSF *xcsf, const char *filename);

void
xcsf_free(struct XCSF *xcsf);

void
xcsf_init(struct XCSF *xcsf);

void
xcsf_print_pset(const struct XCSF *xcsf, const bool print_cond,
                const bool print_act, const bool print_pred);

void
xcsf_ae_to_classifier(struct XCSF *xcsf, const int y_dim, const int n_del);

void
xcsf_pred_expand(const struct XCSF *xcsf);

void
xcsf_retrieve_pset(struct XCSF *xcsf);

void
xcsf_store_pset(struct XCSF *xcsf);
