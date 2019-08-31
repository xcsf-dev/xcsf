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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_softmax.h"
 
void neural_layer_softmax_add(XCSF *xcsf, NET *net, int in, double temp, int p)
{
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = SOFTMAX;
    l->layer_vptr = &layer_softmax_vtbl;
    l->temp = temp;
    l->num_inputs = in;
    l->num_outputs = in;
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = calloc(l->num_inputs, sizeof(double));
    neural_layer_insert(xcsf, net, l, p); 
}

void neural_layer_softmax_copy(XCSF *xcsf, LAYER *to, LAYER *from)
{
    (void)xcsf;
    to->num_inputs = from->num_inputs;
    to->num_outputs = from->num_outputs;
    to->temp = from->temp;
}

void neural_layer_softmax_rand(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}

void neural_layer_softmax_forward(XCSF *xcsf, LAYER *l, double *input)
{
    (void)xcsf;
    double largest = input[0];
    for(int i = 1; i < l->num_inputs; i++) {
        if(input[i] > largest) {
            largest = input[i];
        }
    }
    double sum = 0;
    for(int i = 0; i < l->num_inputs; i++) {
        double e = exp(input[i]/l->temp - largest/l->temp);
        sum += e;
        l->output[i] = e;
    }
    for(int i = 0; i < l->num_inputs; i++) {
        l->output[i] /= sum;
    }                                     
}

void neural_layer_softmax_backward(XCSF *xcsf, LAYER *l, NET *net)
{
    (void)xcsf;
    for(int i = 0; i < l->num_inputs; i++) {
        net->delta[i] += l->delta[i];
    }
}

void neural_layer_softmax_update(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}

void neural_layer_softmax_print(XCSF *xcsf, LAYER *l, _Bool print_weights)
{
    (void)xcsf; (void)print_weights;
    printf("softmax in = %d, out = %d, temp = %f\n", 
            l->num_inputs, l->num_outputs, l->temp);
}

_Bool neural_layer_softmax_mutate(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
    return false;
}

_Bool neural_layer_softmax_crossover(XCSF *xcsf, LAYER *l1, LAYER *l2)
{
    (void)xcsf; (void)l1; (void)l2;
    return false;
}

void neural_layer_softmax_free(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    free(l->output);
    free(l->delta);
}

double* neural_layer_softmax_output(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    return l->output;
}
