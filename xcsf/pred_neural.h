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
 * @file pred_neural.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Multi-layer perceptron neural network prediction functions.
 */

#pragma once

#include "neural.h"
#include "prediction.h"
#include "xcsf.h"

/**
 * @brief Multi-layer perceptron neural network prediction data structure.
 */
typedef struct PRED_NEURAL {
    struct NET net; //!< Neural network
} PRED_NEURAL;

_Bool
pred_neural_crossover(const struct XCSF *xcsf, const struct CL *c1,
                      const struct CL *c2);

_Bool
pred_neural_mutate(const struct XCSF *xcsf, const struct CL *c);

double
pred_neural_eta(const struct XCSF *xcsf, const struct CL *c, const int layer);

int
pred_neural_connections(const struct XCSF *xcsf, const struct CL *c,
                        const int layer);

int
pred_neural_layers(const struct XCSF *xcsf, const struct CL *c);

int
pred_neural_neurons(const struct XCSF *xcsf, const struct CL *c,
                    const int layer);

double
pred_neural_size(const struct XCSF *xcsf, const struct CL *c);

size_t
pred_neural_load(const struct XCSF *xcsf, struct CL *c, FILE *fp);

size_t
pred_neural_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp);

void
pred_neural_compute(const struct XCSF *xcsf, const struct CL *c,
                    const double *x);

void
pred_neural_copy(const struct XCSF *xcsf, struct CL *dest,
                 const struct CL *src);

void
pred_neural_free(const struct XCSF *xcsf, const struct CL *c);

void
pred_neural_init(const struct XCSF *xcsf, struct CL *c);

void
pred_neural_print(const struct XCSF *xcsf, const struct CL *c);

void
pred_neural_update(const struct XCSF *xcsf, const struct CL *c, const double *x,
                   const double *y);

void
pred_neural_expand(const struct XCSF *xcsf, const struct CL *c);

void
pred_neural_ae_to_classifier(const struct XCSF *xcsf, const struct CL *c,
                             const int n_del);

/**
 * @brief Multi-layer perceptron neural network prediction implemented
 * functions.
 */
static struct PredVtbl const pred_neural_vtbl = {
    &pred_neural_crossover, &pred_neural_mutate, &pred_neural_compute,
    &pred_neural_copy,      &pred_neural_free,   &pred_neural_init,
    &pred_neural_print,     &pred_neural_update, &pred_neural_size,
    &pred_neural_save,      &pred_neural_load
};
