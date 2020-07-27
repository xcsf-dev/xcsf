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
 * @brief Functions for setting and printing XCSF parameters.
 */

#pragma once

void param_free(const XCSF *xcsf);
void param_print(const XCSF *xcsf);
void param_init(XCSF *xcsf);
size_t param_load(XCSF *xcsf, FILE *fp);
size_t param_save(const XCSF *xcsf, FILE *fp);
// setters
void param_set_omp_num_threads(XCSF *xcsf, int a);
void param_set_pop_init(XCSF *xcsf, _Bool a);
void param_set_max_trials(XCSF *xcsf, int a);
void param_set_perf_trials(XCSF *xcsf, int a);
void param_set_pop_size(XCSF *xcsf, int a);
void param_set_loss_func(XCSF *xcsf, int a);
void param_set_gamma(XCSF *xcsf, double a);
void param_set_teletransportation(XCSF *xcsf, int a);
void param_set_p_explore(XCSF *xcsf, double a);
void param_set_alpha(XCSF *xcsf, double a);
void param_set_beta(XCSF *xcsf, double a);
void param_set_delta(XCSF *xcsf, double a);
void param_set_eps_0(XCSF *xcsf, double a);
void param_set_err_reduc(XCSF *xcsf, double a);
void param_set_fit_reduc(XCSF *xcsf, double a);
void param_set_init_error(XCSF *xcsf, double a);
void param_set_init_fitness(XCSF *xcsf, double a);
void param_set_nu(XCSF *xcsf, double a);
void param_set_theta_del(XCSF *xcsf, int a);
void param_set_cond_type(XCSF *xcsf, int a);
void param_set_pred_type(XCSF *xcsf, int a);
void param_set_act_type(XCSF *xcsf, int a);
void param_set_m_probation(XCSF *xcsf, int a);
void param_set_sam_type(XCSF *xcsf, int a);
void param_set_p_crossover(XCSF *xcsf, double a);
void param_set_theta_ea(XCSF *xcsf, double a);
void param_set_lambda(XCSF *xcsf, int a);
void param_set_ea_select_type(XCSF *xcsf, int a);
void param_set_ea_select_size(XCSF *xcsf, double a);
void param_set_cond_max(XCSF *xcsf, double a);
void param_set_cond_min(XCSF *xcsf, double a);
void param_set_cond_smin(XCSF *xcsf, double a);
void param_set_cond_bits(XCSF *xcsf, int a);
void param_set_stateful(XCSF *xcsf, _Bool a);
void param_set_max_k(XCSF *xcsf, int a);
void param_set_max_t(XCSF *xcsf, int a);
void param_set_gp_num_cons(XCSF *xcsf, int a);
void param_set_gp_init_depth(XCSF *xcsf, int a);
void param_set_max_neuron_grow(XCSF *xcsf, int a);
void param_set_cond_eta(XCSF *xcsf, double a);
void param_set_cond_evolve_weights(XCSF *xcsf, _Bool a);
void param_set_cond_evolve_neurons(XCSF *xcsf, _Bool a);
void param_set_cond_evolve_functions(XCSF *xcsf, _Bool a);
void param_set_cond_evolve_connectivity(XCSF *xcsf, _Bool a);
void param_set_cond_output_activation(XCSF *xcsf, int a);
void param_set_cond_hidden_activation(XCSF *xcsf, int a);
void param_set_pred_reset(XCSF *xcsf, _Bool a);
void param_set_pred_eta(XCSF *xcsf, double a);
void param_set_pred_x0(XCSF *xcsf, double a);
void param_set_pred_rls_scale_factor(XCSF *xcsf, double a);
void param_set_pred_rls_lambda(XCSF *xcsf, double a);
void param_set_pred_evolve_weights(XCSF *xcsf, _Bool a);
void param_set_pred_evolve_neurons(XCSF *xcsf, _Bool a);
void param_set_pred_evolve_functions(XCSF *xcsf, _Bool a);
void param_set_pred_evolve_connectivity(XCSF *xcsf, _Bool a);
void param_set_pred_evolve_eta(XCSF *xcsf, _Bool a);
void param_set_pred_sgd_weights(XCSF *xcsf, _Bool a);
void param_set_pred_momentum(XCSF *xcsf, double a);
void param_set_pred_output_activation(XCSF *xcsf, int a);
void param_set_pred_hidden_activation(XCSF *xcsf, int a);
void param_set_ea_subsumption(XCSF *xcsf, _Bool a);
void param_set_set_subsumption(XCSF *xcsf, _Bool a);
void param_set_theta_sub(XCSF *xcsf, int a);
void param_set_explore(XCSF *xcsf, _Bool a);
void param_set_x_dim(XCSF *xcsf, int a);
void param_set_y_dim(XCSF *xcsf, int a);
void param_set_n_actions(XCSF *xcsf, int a);
