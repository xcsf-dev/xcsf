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
 * @date 2015--2021.
 * @brief Functions for setting and printing parameters.
 */

#pragma once

#include "loss.h"
#include "xcsf.h"

const char *
param_json_export(const struct XCSF *xcsf);

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

void
param_set_pop_init(struct XCSF *xcsf, const bool a);

void
param_set_max_trials(struct XCSF *xcsf, const int a);

void
param_set_perf_trials(struct XCSF *xcsf, const int a);

void
param_set_pop_size(struct XCSF *xcsf, const int a);

void
param_set_loss_func_string(struct XCSF *xcsf, const char *a);

void
param_set_loss_func(struct XCSF *xcsf, const int a);

void
param_set_stateful(struct XCSF *xcsf, const bool a);

void
param_set_compaction(struct XCSF *xcsf, const bool a);

void
param_set_huber_delta(struct XCSF *xcsf, const double a);

void
param_set_gamma(struct XCSF *xcsf, const double a);

void
param_set_teletransportation(struct XCSF *xcsf, const int a);

void
param_set_p_explore(struct XCSF *xcsf, const double a);

void
param_set_alpha(struct XCSF *xcsf, const double a);

void
param_set_beta(struct XCSF *xcsf, const double a);

void
param_set_delta(struct XCSF *xcsf, const double a);

void
param_set_e0(struct XCSF *xcsf, const double a);

void
param_set_init_error(struct XCSF *xcsf, const double a);

void
param_set_init_fitness(struct XCSF *xcsf, const double a);

void
param_set_nu(struct XCSF *xcsf, const double a);

void
param_set_theta_del(struct XCSF *xcsf, const int a);

void
param_set_m_probation(struct XCSF *xcsf, const int a);

void
param_set_set_subsumption(struct XCSF *xcsf, const bool a);

void
param_set_theta_sub(struct XCSF *xcsf, const int a);

void
param_set_x_dim(struct XCSF *xcsf, const int a);

void
param_set_explore(struct XCSF *xcsf, const bool a);

void
param_set_y_dim(struct XCSF *xcsf, const int a);

void
param_set_n_actions(struct XCSF *xcsf, const int a);
