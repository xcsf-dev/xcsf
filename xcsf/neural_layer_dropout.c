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
#include "neural_layer_dropout.h"

LAYER *neural_layer_dropout_init(XCSF *xcsf, int num_inputs, double probability)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = DROPOUT;
    l->layer_vptr = &layer_dropout_vtbl;
    l->num_inputs = num_inputs;
    l->num_outputs = num_inputs;
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = malloc(l->num_inputs*sizeof(double));
    l->probability = probability;
    l->rand = malloc(l->num_inputs*sizeof(double));
    l->scale = 1./(1.-probability);
    return l;
}

void neural_layer_dropout_copy(XCSF *xcsf, LAYER *to, LAYER *from)
{
    (void)xcsf;
    to->num_inputs = from->num_inputs;
    to->num_outputs = from->num_inputs;
    to->probability = from->probability;
    to->scale = from->scale;
}
 
void neural_layer_dropout_free(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    free(l->output);
    free(l->delta);
    free(l->rand);
}

void neural_layer_dropout_rand(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}
 
void neural_layer_dropout_forward(XCSF *xcsf, LAYER *l, double *input)
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
                l->output[i] = 0;
            }
            else {
                l->output[i] = input[i] * l->scale;
            }
        }
    }
}

void neural_layer_dropout_backward(XCSF *xcsf, LAYER *l, BPN *bpn)
{
    (void)xcsf;
    if(!bpn->delta) {
        return;
    }
    for(int i = 0; i < l->num_inputs; i++) {
        if(l->rand[i] < l->probability) {
            bpn->delta[i] = 0;
        }
        else {
            bpn->delta[i] += l->delta[i] * l->scale;
        }
    }
}

void neural_layer_dropout_update(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}

_Bool neural_layer_dropout_mutate(XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
    return false;
}

double *neural_layer_dropout_output(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    return l->output;
}

void neural_layer_dropout_print(XCSF *xcsf, LAYER *l, _Bool print_weights)
{
    (void)xcsf; (void)l; (void)print_weights;
}
