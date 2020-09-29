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
 * @file neural_layer_maxpool.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D maxpooling layer.
 */

#pragma once

#include "neural_layer.h"
#include "xcsf.h"

struct Layer *
neural_layer_maxpool_init(const struct XCSF *xcsf, const int h, const int w,
                          const int c, const int size, const int stride,
                          const int pad);

struct Layer *
neural_layer_maxpool_copy(const struct XCSF *xcsf, const struct Layer *src);

void
neural_layer_maxpool_rand(const struct XCSF *xcsf, struct Layer *l);

void
neural_layer_maxpool_forward(const struct XCSF *xcsf, const struct Layer *l,
                             const double *input);

void
neural_layer_maxpool_backward(const struct XCSF *xcsf, const struct Layer *l,
                              const double *input, double *delta);

void
neural_layer_maxpool_update(const struct XCSF *xcsf, const struct Layer *l);

void
neural_layer_maxpool_print(const struct XCSF *xcsf, const struct Layer *l,
                           const _Bool print_weights);

_Bool
neural_layer_maxpool_mutate(const struct XCSF *xcsf, struct Layer *l);

void
neural_layer_maxpool_free(const struct XCSF *xcsf, const struct Layer *l);

double *
neural_layer_maxpool_output(const struct XCSF *xcsf, const struct Layer *l);

size_t
neural_layer_maxpool_save(const struct XCSF *xcsf, const struct Layer *l,
                          FILE *fp);

size_t
neural_layer_maxpool_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp);

void
neural_layer_maxpool_resize(const struct XCSF *xcsf, struct Layer *l,
                            const struct Layer *prev);

/**
 * @brief Neural maxpooling layer implemented functions.
 */
static struct LayerVtbl const layer_maxpool_vtbl = {
    &neural_layer_maxpool_mutate,  &neural_layer_maxpool_resize,
    &neural_layer_maxpool_copy,    &neural_layer_maxpool_free,
    &neural_layer_maxpool_rand,    &neural_layer_maxpool_print,
    &neural_layer_maxpool_update,  &neural_layer_maxpool_backward,
    &neural_layer_maxpool_forward, &neural_layer_maxpool_output,
    &neural_layer_maxpool_save,    &neural_layer_maxpool_load
};
