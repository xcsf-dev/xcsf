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
  
void neural_layer_dropout_init(XCSF *xcsf, LAYER *l, int num_inputs, double probability);
void neural_layer_dropout_copy(XCSF *xcsf, LAYER *to, LAYER *from);
void neural_layer_dropout_rand(XCSF *xcsf, LAYER *l);
void neural_layer_dropout_forward(XCSF *xcsf, LAYER *l, double *input);
void neural_layer_dropout_backward(XCSF *xcsf, LAYER *l, BPN *bpn);
void neural_layer_dropout_update(XCSF *xcsf, LAYER *l);
void neural_layer_dropout_print(XCSF *xcsf, LAYER *l, _Bool print_weights);
_Bool neural_layer_dropout_mutate(XCSF *xcsf, LAYER *l);
void neural_layer_dropout_free(XCSF *xcsf, LAYER *l);
double* neural_layer_dropout_output(XCSF *xcsf, LAYER *l);

static struct LayerVtbl const layer_dropout_vtbl = {
    &neural_layer_dropout_mutate,
    &neural_layer_dropout_copy,
    &neural_layer_dropout_free,
    &neural_layer_dropout_rand,
    &neural_layer_dropout_print,
    &neural_layer_dropout_update,
    &neural_layer_dropout_backward,
    &neural_layer_dropout_forward,
    &neural_layer_dropout_output
};