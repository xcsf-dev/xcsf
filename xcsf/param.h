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

#include "loss.h"
#include "xcsf.h"

void
param_free(struct XCSF *xcsf);

void
param_print(const struct XCSF *xcsf);

void
param_init(struct XCSF *xcsf, const int x_dim, const int y_dim,
           const int n_actions);

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
param_set_loss_func_string(struct XCSF *xcsf, const char *a)
{
    xcsf->LOSS_FUNC = loss_type_as_int(a);
    loss_set_func(xcsf);
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
param_set_stateful(struct XCSF *xcsf, const bool a)
{
    xcsf->STATEFUL = a;
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
