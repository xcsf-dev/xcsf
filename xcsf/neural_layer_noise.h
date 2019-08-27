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
  
void neural_layer_noise_init(XCSF *xcsf, NET *net, int ninputs, double p, double s);
void neural_layer_noise_copy(XCSF *xcsf, LAYER *to, LAYER *from);
void neural_layer_noise_rand(XCSF *xcsf, LAYER *l);
void neural_layer_noise_forward(XCSF *xcsf, LAYER *l, double *input);
void neural_layer_noise_backward(XCSF *xcsf, LAYER *l, NET *net);
void neural_layer_noise_update(XCSF *xcsf, LAYER *l);
void neural_layer_noise_print(XCSF *xcsf, LAYER *l, _Bool print_weights);
_Bool neural_layer_noise_mutate(XCSF *xcsf, LAYER *l);
_Bool neural_layer_noise_crossover(XCSF *xcsf, LAYER *l1, LAYER *l2);
void neural_layer_noise_free(XCSF *xcsf, LAYER *l);
double* neural_layer_noise_output(XCSF *xcsf, LAYER *l);

static struct LayerVtbl const layer_noise_vtbl = {
    &neural_layer_noise_crossover,
    &neural_layer_noise_mutate,
    &neural_layer_noise_copy,
    &neural_layer_noise_free,
    &neural_layer_noise_rand,
    &neural_layer_noise_print,
    &neural_layer_noise_update,
    &neural_layer_noise_backward,
    &neural_layer_noise_forward,
    &neural_layer_noise_output
};
