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
 * @file neural_layer_convolutional.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D convolutional layer.
 */

#pragma once

#include "neural_layer.h"
#include "xcsf.h"

LAYER *
neural_layer_convolutional_init(const struct XCSF *xcsf, int h, int w, int c,
                                int n_filters, int kernel_size, int stride,
                                int pad, int f, uint32_t o);

LAYER *
neural_layer_convolutional_copy(const struct XCSF *xcsf, const LAYER *src);

void
neural_layer_convolutional_rand(const struct XCSF *xcsf, LAYER *l);

void
neural_layer_convolutional_forward(const struct XCSF *xcsf, const LAYER *l,
                                   const double *input);

void
neural_layer_convolutional_backward(const struct XCSF *xcsf, const LAYER *l,
                                    const double *input, double *delta);

void
neural_layer_convolutional_update(const struct XCSF *xcsf, const LAYER *l);

void
neural_layer_convolutional_print(const struct XCSF *xcsf, const LAYER *l,
                                 _Bool print_weights);

_Bool
neural_layer_convolutional_mutate(const struct XCSF *xcsf, LAYER *l);

void
neural_layer_convolutional_free(const struct XCSF *xcsf, const LAYER *l);

double *
neural_layer_convolutional_output(const struct XCSF *xcsf, const LAYER *l);

size_t
neural_layer_convolutional_save(const struct XCSF *xcsf, const LAYER *l,
                                FILE *fp);

size_t
neural_layer_convolutional_load(const struct XCSF *xcsf, LAYER *l, FILE *fp);

void
neural_layer_convolutional_resize(const struct XCSF *xcsf, LAYER *l,
                                  const LAYER *prev);

static struct LayerVtbl const layer_convolutional_vtbl = {
    &neural_layer_convolutional_mutate,  &neural_layer_convolutional_resize,
    &neural_layer_convolutional_copy,    &neural_layer_convolutional_free,
    &neural_layer_convolutional_rand,    &neural_layer_convolutional_print,
    &neural_layer_convolutional_update,  &neural_layer_convolutional_backward,
    &neural_layer_convolutional_forward, &neural_layer_convolutional_output,
    &neural_layer_convolutional_save,    &neural_layer_convolutional_load,
};
