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
 * @file rule_network.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Neural network rule (condition + prediction) functions.
 */

#pragma once

/**
 * @brief Neural network condition-prediction data structure.
 */
typedef struct RULE_NETWORK {
    NET net; //!< Neural network
} RULE_NETWORK;

_Bool rule_network_cond_crossover(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool rule_network_cond_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool rule_network_cond_match(const XCSF *xcsf, const CL *c, const double *x);
_Bool rule_network_cond_mutate(const XCSF *xcsf, const CL *c);
void rule_network_cond_copy(const XCSF *xcsf, CL *dest, const CL *src);
void rule_network_cond_cover(const XCSF *xcsf, const CL *c, const double *x);
void rule_network_cond_free(const XCSF *xcsf, const CL *c);
void rule_network_cond_init(const XCSF *xcsf, CL *c);
void rule_network_cond_print(const XCSF *xcsf, const CL *c);
void rule_network_cond_update(const XCSF *xcsf, const CL *c, const double *x,
                              const double *y);
int rule_network_cond_size(const XCSF *xcsf, const CL *c);
size_t rule_network_cond_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t rule_network_cond_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Neural network rule condition implemented functions.
 */
static struct CondVtbl const rule_network_cond_vtbl = {
    &rule_network_cond_crossover,
    &rule_network_cond_general,
    &rule_network_cond_match,
    &rule_network_cond_mutate,
    &rule_network_cond_copy,
    &rule_network_cond_cover,
    &rule_network_cond_free,
    &rule_network_cond_init,
    &rule_network_cond_print,
    &rule_network_cond_update,
    &rule_network_cond_size,
    &rule_network_cond_save,
    &rule_network_cond_load
};

_Bool rule_network_pred_crossover(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool rule_network_pred_mutate(const XCSF *xcsf, const CL *c);
double rule_network_eta(const XCSF *xcsf, const CL *c, int layer);
int rule_network_layers(const XCSF *xcsf, const CL *c);
int rule_network_neurons(const XCSF *xcsf, const CL *c, int layer);
int rule_network_pred_size(const XCSF *xcsf, const CL *c);
size_t rule_network_pred_load(const XCSF *xcsf, CL *c, FILE *fp);
size_t rule_network_pred_save(const XCSF *xcsf, const CL *c, FILE *fp);
void rule_network_pred_compute(const XCSF *xcsf, const CL *c, const double *x);
void rule_network_pred_copy(const XCSF *xcsf, CL *dest, const CL *src);
void rule_network_pred_free(const XCSF *xcsf, const CL *c);
void rule_network_pred_init(const XCSF *xcsf, CL *c);
void rule_network_pred_print(const XCSF *xcsf, const CL *c);
void rule_network_pred_update(const XCSF *xcsf, const CL *c, const double *x,
                              const double *y);

/**
 * @brief Multi-layer perceptron neural network prediction implemented functions.
 */
static struct PredVtbl const rule_network_pred_vtbl = {
    &rule_network_pred_crossover,
    &rule_network_pred_mutate,
    &rule_network_pred_compute,
    &rule_network_pred_copy,
    &rule_network_pred_free,
    &rule_network_pred_init,
    &rule_network_pred_print,
    &rule_network_pred_update,
    &rule_network_pred_size,
    &rule_network_pred_save,
    &rule_network_pred_load
};
