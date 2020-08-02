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
param_set_omp_num_threads(struct XCSF *xcsf, int a);

void
param_set_pop_init(struct XCSF *xcsf, _Bool a);

void
param_set_max_trials(struct XCSF *xcsf, int a);

void
param_set_perf_trials(struct XCSF *xcsf, int a);

void
param_set_pop_size(struct XCSF *xcsf, int a);

void
param_set_loss_func(struct XCSF *xcsf, int a);

void
param_set_gamma(struct XCSF *xcsf, double a);

void
param_set_teletransportation(struct XCSF *xcsf, int a);

void
param_set_p_explore(struct XCSF *xcsf, double a);

void
param_set_alpha(struct XCSF *xcsf, double a);

void
param_set_beta(struct XCSF *xcsf, double a);

void
param_set_delta(struct XCSF *xcsf, double a);

void
param_set_eps_0(struct XCSF *xcsf, double a);

void
param_set_err_reduc(struct XCSF *xcsf, double a);

void
param_set_fit_reduc(struct XCSF *xcsf, double a);

void
param_set_init_error(struct XCSF *xcsf, double a);

void
param_set_init_fitness(struct XCSF *xcsf, double a);

void
param_set_nu(struct XCSF *xcsf, double a);

void
param_set_theta_del(struct XCSF *xcsf, int a);

void
param_set_cond_type(struct XCSF *xcsf, int a);

void
param_set_pred_type(struct XCSF *xcsf, int a);

void
param_set_act_type(struct XCSF *xcsf, int a);

void
param_set_m_probation(struct XCSF *xcsf, int a);

void
param_set_sam_type(struct XCSF *xcsf, int a);

void
param_set_p_crossover(struct XCSF *xcsf, double a);

void
param_set_theta_ea(struct XCSF *xcsf, double a);

void
param_set_lambda(struct XCSF *xcsf, int a);

void
param_set_ea_select_type(struct XCSF *xcsf, int a);

void
param_set_ea_select_size(struct XCSF *xcsf, double a);

void
param_set_cond_max(struct XCSF *xcsf, double a);

void
param_set_cond_min(struct XCSF *xcsf, double a);

void
param_set_cond_smin(struct XCSF *xcsf, double a);

void
param_set_cond_bits(struct XCSF *xcsf, int a);

void
param_set_stateful(struct XCSF *xcsf, _Bool a);

void
param_set_max_k(struct XCSF *xcsf, int a);

void
param_set_max_t(struct XCSF *xcsf, int a);

void
param_set_gp_num_cons(struct XCSF *xcsf, int a);

void
param_set_gp_init_depth(struct XCSF *xcsf, int a);

void
param_set_max_neuron_grow(struct XCSF *xcsf, int a);

void
param_set_cond_eta(struct XCSF *xcsf, double a);

void
param_set_cond_evolve_weights(struct XCSF *xcsf, _Bool a);

void
param_set_cond_evolve_neurons(struct XCSF *xcsf, _Bool a);

void
param_set_cond_evolve_functions(struct XCSF *xcsf, _Bool a);

void
param_set_cond_evolve_connectivity(struct XCSF *xcsf, _Bool a);

void
param_set_cond_output_activation(struct XCSF *xcsf, int a);

void
param_set_cond_hidden_activation(struct XCSF *xcsf, int a);

void
param_set_pred_reset(struct XCSF *xcsf, _Bool a);

void
param_set_pred_eta(struct XCSF *xcsf, double a);

void
param_set_pred_x0(struct XCSF *xcsf, double a);

void
param_set_pred_rls_scale_factor(struct XCSF *xcsf, double a);

void
param_set_pred_rls_lambda(struct XCSF *xcsf, double a);

void
param_set_pred_evolve_weights(struct XCSF *xcsf, _Bool a);

void
param_set_pred_evolve_neurons(struct XCSF *xcsf, _Bool a);

void
param_set_pred_evolve_functions(struct XCSF *xcsf, _Bool a);

void
param_set_pred_evolve_connectivity(struct XCSF *xcsf, _Bool a);

void
param_set_pred_evolve_eta(struct XCSF *xcsf, _Bool a);

void
param_set_pred_sgd_weights(struct XCSF *xcsf, _Bool a);

void
param_set_pred_momentum(struct XCSF *xcsf, double a);

void
param_set_pred_output_activation(struct XCSF *xcsf, int a);

void
param_set_pred_hidden_activation(struct XCSF *xcsf, int a);

void
param_set_ea_subsumption(struct XCSF *xcsf, _Bool a);

void
param_set_set_subsumption(struct XCSF *xcsf, _Bool a);

void
param_set_theta_sub(struct XCSF *xcsf, int a);

void
param_set_explore(struct XCSF *xcsf, _Bool a);

void
param_set_x_dim(struct XCSF *xcsf, int a);

void
param_set_y_dim(struct XCSF *xcsf, int a);

void
param_set_n_actions(struct XCSF *xcsf, int a);
