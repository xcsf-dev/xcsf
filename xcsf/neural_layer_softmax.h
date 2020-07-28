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
 * @file neural_layer_softmax.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a softmax layer.
 */

#pragma once

LAYER *neural_layer_softmax_init(const XCSF *xcsf, int in, double temp);
LAYER *neural_layer_softmax_copy(const XCSF *xcsf, const LAYER *src);
void neural_layer_softmax_rand(const XCSF *xcsf, LAYER *l);
void neural_layer_softmax_forward(const XCSF *xcsf, const LAYER *l, const double *input);
void neural_layer_softmax_backward(const XCSF *xcsf, const LAYER *l, const double *input,
                                   double *delta);
void neural_layer_softmax_update(const XCSF *xcsf, const LAYER *l);
void neural_layer_softmax_print(const XCSF *xcsf, const LAYER *l, _Bool print_weights);
_Bool neural_layer_softmax_mutate(const XCSF *xcsf, LAYER *l);
void neural_layer_softmax_free(const XCSF *xcsf, const LAYER *l);
double *neural_layer_softmax_output(const XCSF *xcsf, const LAYER *l);
size_t neural_layer_softmax_save(const XCSF *xcsf, const LAYER *l, FILE *fp);
size_t neural_layer_softmax_load(const XCSF *xcsf, LAYER *l, FILE *fp);
void neural_layer_softmax_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev);

static struct LayerVtbl const layer_softmax_vtbl = {
    &neural_layer_softmax_mutate,
    &neural_layer_softmax_resize,
    &neural_layer_softmax_copy,
    &neural_layer_softmax_free,
    &neural_layer_softmax_rand,
    &neural_layer_softmax_print,
    &neural_layer_softmax_update,
    &neural_layer_softmax_backward,
    &neural_layer_softmax_forward,
    &neural_layer_softmax_output,
    &neural_layer_softmax_save,
    &neural_layer_softmax_load
};
