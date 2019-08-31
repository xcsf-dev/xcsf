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
#include "neural_layer_noise.h"

void neural_layer_noise_add(XCSF *xcsf, NET *net, int in, double prob, double std, int p)
{
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = NOISE;
    l->layer_vptr = &layer_noise_vtbl;
    l->num_inputs = in;
    l->num_outputs = in;
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = malloc(l->num_inputs*sizeof(double));
    l->probability = prob;
    l->scale = std;
    l->rand = malloc(l->num_inputs*sizeof(double));
    neural_layer_insert(xcsf, net, l, p); 
}

void neural_layer_noise_copy(XCSF *xcsf, LAYER *to, LAYER *from)
{
    (void)xcsf;
    to->num_inputs = from->num_inputs;
    to->num_outputs = from->num_outputs;
    to->probability = from->probability;
    to->scale = from->scale;
}
 
void neural_layer_noise_free(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    free(l->output);
    free(l->delta);
    free(l->rand);
}

void neural_layer_noise_rand(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}
 
void neural_layer_noise_forward(XCSF *xcsf, LAYER *l, double *input)
{
    if(!xcsf->train) {
        for(int i = 0; i < l->num_inputs; i++) {
            l->output[i] = input[i];
        }
    }
    else {
        for(int i = 0; i < l->num_inputs; i++) {
            l->rand[i] = rand_uniform(0,1);
            if(l->rand[i] < l->probability) {
                l->output[i] = input[i] + rand_normal(0, l->scale);
            }
            else {
                l->output[i] = input[i];
            }
        }
    }
}

void neural_layer_noise_backward(XCSF *xcsf, LAYER *l, NET *net)
{
    (void)xcsf;
    if(!net->delta) {
        return;
    }
    for(int i = 0; i < l->num_inputs; i++) {
        net->delta[i] += l->delta[i];
    }
}

void neural_layer_noise_update(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}

_Bool neural_layer_noise_mutate(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
    return false;
}

_Bool neural_layer_noise_crossover(XCSF *xcsf, LAYER *l1, LAYER *l2)
{
    (void)xcsf; (void)l1; (void)l2;
    return false;
}

double *neural_layer_noise_output(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    return l->output;
}

void neural_layer_noise_print(XCSF *xcsf, LAYER *l, _Bool print_weights)
{
    (void)xcsf; (void)print_weights;
    printf("noise nin = %d, out = %d, prob = %f, stdev = %f\n",
            l->num_inputs, l->num_outputs, l->probability, l->scale);
}
