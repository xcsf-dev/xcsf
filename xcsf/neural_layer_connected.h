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
  
void neural_layer_connected_init(LAYER *l, int num_inputs, int num_outputs, int activation);
void neural_layer_connected_copy(LAYER *to, LAYER *from);
void neural_layer_connected_rand(LAYER *l);
void neural_layer_connected_forward(LAYER *l, double *input);
void neural_layer_connected_backward(LAYER *l, BPN *bpn);
void neural_layer_connected_update(XCSF *xcsf, LAYER *l);
void neural_layer_connected_print(LAYER *l, _Bool print_weights);
_Bool neural_layer_connected_mutate(XCSF *xcsf, LAYER *l);
void neural_layer_connected_free(LAYER *l);
double* neural_layer_connected_output(LAYER *l);

static struct LayerVtbl const layer_connected_vtbl = {
    &neural_layer_connected_init,
    &neural_layer_connected_mutate,
    &neural_layer_connected_copy,
    &neural_layer_connected_free,
    &neural_layer_connected_rand,
    &neural_layer_connected_print,
    &neural_layer_connected_update,
    &neural_layer_connected_backward,
    &neural_layer_connected_forward,
    &neural_layer_connected_output
};
