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
 * @file param.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Functions for setting and printing parameters.
 */

#pragma once

#include "gp.h"
#include "loss.h"
#include "xcsf.h"

void
param_free(const struct XCSF *xcsf);

void
param_print(const struct XCSF *xcsf);

void
param_init(struct XCSF *xcsf);

size_t
param_load(struct XCSF *xcsf, FILE *fp);

size_t
param_save(const struct XCSF *xcsf, FILE *fp);

/* setters */

void
param_set_omp_num_threads(struct XCSF *xcsf, const int a);

static inline void
param_set_pop_init(struct XCSF *xcsf, const bool a)
{
    xcsf->POP_INIT = a;
}

static inline void
param_set_max_trials(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set MAX_TRIALS too small\n");
        xcsf->MAX_TRIALS = 1;
    } else {
        xcsf->MAX_TRIALS = a;
    }
}

static inline void
param_set_perf_trials(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set PERF_TRIALS too small\n");
        xcsf->PERF_TRIALS = 1;
    } else {
        xcsf->PERF_TRIALS = a;
    }
}

static inline void
param_set_pop_size(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set POP_SIZE too small\n");
        xcsf->POP_SIZE = 1;
    } else {
        xcsf->POP_SIZE = a;
    }
}

static inline void
param_set_loss_func(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set LOSS_FUNC too small\n");
        xcsf->LOSS_FUNC = 0;
    } else if (a >= LOSS_NUM) {
        printf("Warning: tried to set LOSS_FUNC too large\n");
        xcsf->LOSS_FUNC = LOSS_NUM - 1;
    } else {
        xcsf->LOSS_FUNC = a;
    }
    loss_set_func(xcsf);
}

static inline void
param_set_huber_delta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set HUBER_DELTA too small\n");
        xcsf->HUBER_DELTA = 0;
    } else {
        xcsf->HUBER_DELTA = a;
    }
}

static inline void
param_set_gamma(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set GAMMA too small\n");
        xcsf->GAMMA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set GAMMA too large\n");
        xcsf->GAMMA = 1;
    } else {
        xcsf->GAMMA = a;
    }
}

static inline void
param_set_teletransportation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set TELETRANSPORTATION too small\n");
        xcsf->TELETRANSPORTATION = 0;
    } else {
        xcsf->TELETRANSPORTATION = a;
    }
}

static inline void
param_set_p_explore(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set P_EXPLORE too small\n");
        xcsf->P_EXPLORE = 0;
    } else if (a > 1) {
        printf("Warning: tried to set P_EXPLORE too large\n");
        xcsf->P_EXPLORE = 1;
    } else {
        xcsf->P_EXPLORE = a;
    }
}

static inline void
param_set_alpha(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set ALPHA too small\n");
        xcsf->ALPHA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set ALPHA too large\n");
        xcsf->ALPHA = 1;
    } else {
        xcsf->ALPHA = a;
    }
}

static inline void
param_set_beta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set BETA too small\n");
        xcsf->BETA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set BETA too large\n");
        xcsf->BETA = 1;
    } else {
        xcsf->BETA = a;
    }
}

static inline void
param_set_delta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set DELTA too small\n");
        xcsf->DELTA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set DELTA too large\n");
        xcsf->DELTA = 1;
    } else {
        xcsf->DELTA = a;
    }
}

static inline void
param_set_eps_0(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set EPS_0 too small\n");
        xcsf->EPS_0 = 0;
    } else {
        xcsf->EPS_0 = a;
    }
}

static inline void
param_set_err_reduc(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set ERR_REDUC too small\n");
        xcsf->ERR_REDUC = 0;
    } else if (a > 1) {
        printf("Warning: tried to set ERR_REDUC too large\n");
        xcsf->ERR_REDUC = 1;
    } else {
        xcsf->ERR_REDUC = a;
    }
}

static inline void
param_set_fit_reduc(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set FIT_REDUC too small\n");
        xcsf->FIT_REDUC = 0;
    } else if (a > 1) {
        printf("Warning: tried to set FIT_REDUC too large\n");
        xcsf->FIT_REDUC = 1;
    } else {
        xcsf->FIT_REDUC = a;
    }
}

static inline void
param_set_init_error(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set INIT_ERROR too small\n");
        xcsf->INIT_ERROR = 0;
    } else {
        xcsf->INIT_ERROR = a;
    }
}

static inline void
param_set_init_fitness(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set INIT_FITNESS too small\n");
        xcsf->INIT_FITNESS = 0;
    } else {
        xcsf->INIT_FITNESS = a;
    }
}

static inline void
param_set_nu(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set NU too small\n");
        xcsf->NU = 0;
    } else {
        xcsf->NU = a;
    }
}

static inline void
param_set_theta_del(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set THETA_DEL too small\n");
        xcsf->THETA_DEL = 0;
    } else {
        xcsf->THETA_DEL = a;
    }
}

static inline void
param_set_cond_type(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set COND_TYPE too small\n");
        xcsf->COND_TYPE = 0;
    } else {
        xcsf->COND_TYPE = a;
    }
}

static inline void
param_set_pred_type(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED_TYPE too small\n");
        xcsf->PRED_TYPE = 0;
    } else {
        xcsf->PRED_TYPE = a;
    }
}

static inline void
param_set_act_type(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set ACT_TYPE too small\n");
        xcsf->ACT_TYPE = 0;
    } else {
        xcsf->ACT_TYPE = a;
    }
}

static inline void
param_set_m_probation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set M_PROBATION too small\n");
        xcsf->M_PROBATION = 0;
    } else {
        xcsf->M_PROBATION = a;
    }
}

static inline void
param_set_p_crossover(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set P_CROSSOVER too small\n");
        xcsf->P_CROSSOVER = 0;
    } else if (a > 1) {
        printf("Warning: tried to set P_CROSSOVER too large\n");
        xcsf->P_CROSSOVER = 1;
    } else {
        xcsf->P_CROSSOVER = a;
    }
}

static inline void
param_set_theta_ea(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set THETA_EA too small\n");
        xcsf->THETA_EA = 0;
    } else {
        xcsf->THETA_EA = a;
    }
}

static inline void
param_set_lambda(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set LAMBDA too small\n");
        xcsf->LAMBDA = 0;
    } else {
        xcsf->LAMBDA = a;
    }
}

static inline void
param_set_ea_select_type(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set EA_SELECT_TYPE too small\n");
        xcsf->EA_SELECT_TYPE = 0;
    } else {
        xcsf->EA_SELECT_TYPE = a;
    }
}

static inline void
param_set_ea_select_size(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set EA_SELECT_SIZE too small\n");
        xcsf->EA_SELECT_SIZE = 0;
    } else if (a > 1) {
        printf("Warning: tried to set EA_SELECT_SIZE too large\n");
        xcsf->EA_SELECT_SIZE = 1;
    } else {
        xcsf->EA_SELECT_SIZE = a;
    }
}

static inline void
param_set_cond_max(struct XCSF *xcsf, const double a)
{
    xcsf->COND_MAX = a;
}

static inline void
param_set_cond_min(struct XCSF *xcsf, const double a)
{
    xcsf->COND_MIN = a;
}

static inline void
param_set_cond_smin(struct XCSF *xcsf, const double a)
{
    xcsf->COND_SMIN = a;
}

static inline void
param_set_cond_bits(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set COND_BITS too small\n");
        xcsf->COND_BITS = 1;
    } else {
        xcsf->COND_BITS = a;
    }
}

static inline void
param_set_stateful(struct XCSF *xcsf, const bool a)
{
    xcsf->STATEFUL = a;
}

static inline void
param_set_max_k(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set MAX_K too small\n");
        xcsf->MAX_K = 1;
    } else {
        xcsf->MAX_K = a;
    }
}

static inline void
param_set_max_t(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set MAX_T too small\n");
        xcsf->MAX_T = 1;
    } else {
        xcsf->MAX_T = a;
    }
}

static inline void
param_set_gp_num_cons(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP_NUM_CONS too small\n");
        xcsf->GP_NUM_CONS = 1;
    } else {
        xcsf->GP_NUM_CONS = a;
    }
    if (xcsf->gp_cons != NULL) {
        tree_free_cons(xcsf);
    }
    tree_init_cons(xcsf);
}

static inline void
param_set_gp_init_depth(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP_INIT_DEPTH too small\n");
        xcsf->GP_INIT_DEPTH = 1;
    } else {
        xcsf->GP_INIT_DEPTH = a;
    }
}

static inline void
param_set_max_neuron_grow(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set MAX_NEURON_GROW too small\n");
        xcsf->MAX_NEURON_GROW = 1;
    } else {
        xcsf->MAX_NEURON_GROW = a;
    }
}

static inline void
param_set_cond_eta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set COND_ETA too small\n");
        xcsf->COND_ETA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set COND_ETA too large\n");
        xcsf->COND_ETA = 1;
    } else {
        xcsf->COND_ETA = a;
    }
}

static inline void
param_set_cond_evolve_weights(struct XCSF *xcsf, const bool a)
{
    xcsf->COND_EVOLVE_WEIGHTS = a;
}

static inline void
param_set_cond_evolve_neurons(struct XCSF *xcsf, const bool a)
{
    xcsf->COND_EVOLVE_NEURONS = a;
}

static inline void
param_set_cond_evolve_functions(struct XCSF *xcsf, const bool a)
{
    xcsf->COND_EVOLVE_FUNCTIONS = a;
}

static inline void
param_set_cond_evolve_connectivity(struct XCSF *xcsf, const bool a)
{
    xcsf->COND_EVOLVE_CONNECTIVITY = a;
}

static inline void
param_set_cond_output_activation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set COND_OUTPUT_ACTIVATION too small\n");
        xcsf->COND_OUTPUT_ACTIVATION = 0;
    } else {
        xcsf->COND_OUTPUT_ACTIVATION = a;
    }
}

static inline void
param_set_cond_hidden_activation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set COND_HIDDEN_ACTIVATION too small\n");
        xcsf->COND_HIDDEN_ACTIVATION = 0;
    } else {
        xcsf->COND_HIDDEN_ACTIVATION = a;
    }
}

static inline void
param_set_pred_reset(struct XCSF *xcsf, const bool a)
{
    xcsf->PRED_RESET = a;
}

static inline void
param_set_pred_eta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED_ETA too small\n");
        xcsf->PRED_ETA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set PRED_ETA too large\n");
        xcsf->PRED_ETA = 1;
    } else {
        xcsf->PRED_ETA = a;
    }
}

static inline void
param_set_pred_x0(struct XCSF *xcsf, const double a)
{
    xcsf->PRED_X0 = a;
}

static inline void
param_set_pred_rls_lambda(struct XCSF *xcsf, const double a)
{
    if (a < DBL_EPSILON) {
        printf("Warning: tried to set PRED_RLS_LAMBDA too small\n");
        xcsf->PRED_RLS_LAMBDA = DBL_EPSILON;
    } else {
        xcsf->PRED_RLS_LAMBDA = a;
    }
}

static inline void
param_set_pred_rls_scale_factor(struct XCSF *xcsf, const double a)
{
    xcsf->PRED_RLS_SCALE_FACTOR = a;
}

static inline void
param_set_pred_evolve_weights(struct XCSF *xcsf, const bool a)
{
    xcsf->PRED_EVOLVE_WEIGHTS = a;
}

static inline void
param_set_pred_evolve_neurons(struct XCSF *xcsf, const bool a)
{
    xcsf->PRED_EVOLVE_NEURONS = a;
}

static inline void
param_set_pred_evolve_functions(struct XCSF *xcsf, const bool a)
{
    xcsf->PRED_EVOLVE_FUNCTIONS = a;
}

static inline void
param_set_pred_evolve_connectivity(struct XCSF *xcsf, const bool a)
{
    xcsf->PRED_EVOLVE_CONNECTIVITY = a;
}

static inline void
param_set_pred_evolve_eta(struct XCSF *xcsf, const bool a)
{
    xcsf->PRED_EVOLVE_ETA = a;
}

static inline void
param_set_pred_sgd_weights(struct XCSF *xcsf, const bool a)
{
    xcsf->PRED_SGD_WEIGHTS = a;
}

static inline void
param_set_pred_momentum(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED_MOMENTUM too small\n");
        xcsf->PRED_MOMENTUM = 0;
    } else if (a > 1) {
        printf("Warning: tried to set PRED_MOMENTUM too large\n");
        xcsf->PRED_MOMENTUM = 1;
    } else {
        xcsf->PRED_MOMENTUM = a;
    }
}

static inline void
param_set_pred_decay(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED_DECAY too small\n");
        xcsf->PRED_DECAY = 0;
    } else if (a > 1) {
        printf("Warning: tried to set PRED_DECAY too large\n");
        xcsf->PRED_DECAY = 1;
    } else {
        xcsf->PRED_DECAY = a;
    }
}

static inline void
param_set_pred_output_activation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED_OUTPUT_ACTIVATION too small\n");
        xcsf->PRED_OUTPUT_ACTIVATION = 0;
    } else {
        xcsf->PRED_OUTPUT_ACTIVATION = a;
    }
}

static inline void
param_set_pred_hidden_activation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED_HIDDEN_ACTIVATION too small\n");
        xcsf->PRED_HIDDEN_ACTIVATION = 0;
    } else {
        xcsf->PRED_HIDDEN_ACTIVATION = a;
    }
}

static inline void
param_set_ea_subsumption(struct XCSF *xcsf, const bool a)
{
    xcsf->EA_SUBSUMPTION = a;
}

static inline void
param_set_set_subsumption(struct XCSF *xcsf, const bool a)
{
    xcsf->SET_SUBSUMPTION = a;
}

static inline void
param_set_theta_sub(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set THETA_SUB too small\n");
        xcsf->THETA_SUB = 0;
    } else {
        xcsf->THETA_SUB = a;
    }
}

static inline void
param_set_x_dim(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set x_dim too small\n");
        xcsf->x_dim = 1;
    } else {
        xcsf->x_dim = a;
    }
}

static inline void
param_set_explore(struct XCSF *xcsf, const bool a)
{
    xcsf->explore = a;
}

static inline void
param_set_y_dim(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set y_dim too small\n");
        xcsf->y_dim = 1;
    } else {
        xcsf->y_dim = a;
    }
}

static inline void
param_set_n_actions(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set n_actions too small\n");
        xcsf->n_actions = 1;
    } else {
        xcsf->n_actions = a;
    }
}
