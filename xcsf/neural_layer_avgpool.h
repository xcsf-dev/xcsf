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
 * @file neural_layer_avgpool.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of an average pooling layer.
 */

#pragma once

#include "neural_layer.h"
#include "xcsf.h"

struct Layer *
neural_layer_avgpool_init(const struct XCSF *xcsf, const int h, const int w,
                          const int c);

struct Layer *
neural_layer_avgpool_copy(const struct XCSF *xcsf, const struct Layer *src);

void
neural_layer_avgpool_rand(const struct XCSF *xcsf, struct Layer *l);

void
neural_layer_avgpool_forward(const struct XCSF *xcsf, const struct Layer *l,
                             const double *input);

void
neural_layer_avgpool_backward(const struct XCSF *xcsf, const struct Layer *l,
                              const double *input, double *delta);

void
neural_layer_avgpool_update(const struct XCSF *xcsf, const struct Layer *l);

void
neural_layer_avgpool_print(const struct XCSF *xcsf, const struct Layer *l,
                           const _Bool print_weights);

_Bool
neural_layer_avgpool_mutate(const struct XCSF *xcsf, struct Layer *l);

void
neural_layer_avgpool_free(const struct XCSF *xcsf, const struct Layer *l);

double *
neural_layer_avgpool_output(const struct XCSF *xcsf, const struct Layer *l);

size_t
neural_layer_avgpool_save(const struct XCSF *xcsf, const struct Layer *l,
                          FILE *fp);

size_t
neural_layer_avgpool_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp);

void
neural_layer_avgpool_resize(const struct XCSF *xcsf, struct Layer *l,
                            const struct Layer *prev);

/**
 * @brief Neural average pooling layer implemented functions.
 */
static struct LayerVtbl const layer_avgpool_vtbl = {
    &neural_layer_avgpool_mutate,  &neural_layer_avgpool_resize,
    &neural_layer_avgpool_copy,    &neural_layer_avgpool_free,
    &neural_layer_avgpool_rand,    &neural_layer_avgpool_print,
    &neural_layer_avgpool_update,  &neural_layer_avgpool_backward,
    &neural_layer_avgpool_forward, &neural_layer_avgpool_output,
    &neural_layer_avgpool_save,    &neural_layer_avgpool_load
};
