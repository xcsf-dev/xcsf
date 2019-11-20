/*
 * Copyright (C) 2016--2019 Richard Preen <rpreen@gmail.com>
 *
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
 *
 */
#pragma once

LAYER *neural_layer_softmax_init(XCSF *xcsf, int in, double temp);
LAYER *neural_layer_softmax_copy(XCSF *xcsf, LAYER *from);
void neural_layer_softmax_rand(XCSF *xcsf, LAYER *l);
void neural_layer_softmax_forward(XCSF *xcsf, LAYER *l, double *input);
void neural_layer_softmax_backward(XCSF *xcsf, LAYER *l, NET *net);
void neural_layer_softmax_update(XCSF *xcsf, LAYER *l);
void neural_layer_softmax_print(XCSF *xcsf, LAYER *l, _Bool print_weights);
_Bool neural_layer_softmax_mutate(XCSF *xcsf, LAYER *l);
void neural_layer_softmax_free(XCSF *xcsf, LAYER *l);
double* neural_layer_softmax_output(XCSF *xcsf, LAYER *l);
size_t neural_layer_softmax_save(XCSF *xcsf, LAYER *l, FILE *fp);
size_t neural_layer_softmax_load(XCSF *xcsf, LAYER *l, FILE *fp);

static struct LayerVtbl const layer_softmax_vtbl = {
    &neural_layer_softmax_mutate,
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
