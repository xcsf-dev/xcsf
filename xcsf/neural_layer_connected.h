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
 * @file neural_layer_connected.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2019.
 * @brief An implementation of a fully-connected layer of perceptrons.
 */ 
 
#pragma once

LAYER *neural_layer_connected_init(XCSF *xcsf, int in, int n_init, int n_max, int f, uint32_t o);
LAYER *neural_layer_connected_copy(XCSF *xcsf, LAYER *from);
void neural_layer_connected_rand(XCSF *xcsf, LAYER *l);
void neural_layer_connected_forward(XCSF *xcsf, LAYER *l, double *input);
void neural_layer_connected_backward(XCSF *xcsf, LAYER *l, NET *net);
void neural_layer_connected_update(XCSF *xcsf, LAYER *l);
void neural_layer_connected_print(XCSF *xcsf, LAYER *l, _Bool print_weights);
_Bool neural_layer_connected_mutate(XCSF *xcsf, LAYER *l, LAYER *prev);
void neural_layer_connected_free(XCSF *xcsf, LAYER *l);
double* neural_layer_connected_output(XCSF *xcsf, LAYER *l);
size_t neural_layer_connected_save(XCSF *xcsf, LAYER *l, FILE *fp);
size_t neural_layer_connected_load(XCSF *xcsf, LAYER *l, FILE *fp);

static struct LayerVtbl const layer_connected_vtbl = {
    &neural_layer_connected_mutate,
    &neural_layer_connected_copy,
    &neural_layer_connected_free,
    &neural_layer_connected_rand,
    &neural_layer_connected_print,
    &neural_layer_connected_update,
    &neural_layer_connected_backward,
    &neural_layer_connected_forward,
    &neural_layer_connected_output,
    &neural_layer_connected_save,
    &neural_layer_connected_load,
};
