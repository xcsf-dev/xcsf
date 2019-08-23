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
#include "neural_layer_connected.h"

void neural_init(XCSF *xcsf, BPN *bpn, int num_layers, int *neurons, int *activations)
{
    (void)xcsf;
    bpn->num_layers = num_layers-1; // number of hidden and output layers
    bpn->num_inputs = neurons[0];
    bpn->num_outputs = neurons[num_layers-1];
    bpn->layers = malloc(bpn->num_layers*sizeof(LAYER));
    for(int i = 0; i < bpn->num_layers; i++) {
        neural_layer_init(&bpn->layers[i], CONNECTED, neurons[i], neurons[i+1], activations[i]);
    }
}

void neural_copy(XCSF *xcsf, BPN *to, BPN *from)
{
    (void)xcsf;
    to->num_layers = from->num_layers;
    to->num_outputs = from->num_outputs;
    to->num_inputs = from->num_inputs;
    for(int i = 0; i < from->num_layers; i++) {
        layer_copy(&to->layers[i], &from->layers[i]);
    }
}

void neural_free(XCSF *xcsf, BPN *bpn)
{
    (void)xcsf;
    for(int i = 0; i < bpn->num_layers; i++) {
        layer_free(&bpn->layers[i]);
    }
    free(bpn->layers);
}
 
void neural_rand(XCSF *xcsf, BPN *bpn)
{
    (void)xcsf;
    for(int i = 0; i < bpn->num_layers; i++) {
        layer_rand(&bpn->layers[i]);
    }
}    

_Bool neural_mutate(XCSF *xcsf, BPN *bpn)
{
    _Bool mod = false;
    for(int i = 0; i < bpn->num_layers; i++) {
        if(layer_mutate(xcsf, &bpn->layers[i])) {
            mod = true;
        }
    }
    return mod;
}

void neural_propagate(XCSF *xcsf, BPN *bpn, double *input)
{
    (void)xcsf;
    for(int i = 0; i < bpn->num_layers; i++) {
        layer_forward(&bpn->layers[i], input);
        input = layer_output(&bpn->layers[i]);
    }
}

void neural_learn(XCSF *xcsf, BPN *bpn, double *truth, double *input)
{
    /* reset deltas */
    for(int i = 0; i < bpn->num_layers; i++) {
        LAYER *l = &bpn->layers[i];
        for(int j = 0; j < l->num_outputs; j++) {
            l->delta[j] = 0.0;
        }
    }

    // calculate output layer error
    LAYER *p = &bpn->layers[bpn->num_layers-1];
    for(int i = 0; i < p->num_outputs; i++) {
        p->delta[i] = (truth[i] - p->output[i]);
    }

    /* backward phase */
    for(int i = bpn->num_layers-1; i >= 0; i--) {
        LAYER *l = &bpn->layers[i];
        if(i == 0) {
            bpn->input = input;
            bpn->delta = 0;
        }
        else {
            LAYER *prev = &bpn->layers[i-1];
            bpn->input = prev->output;
            bpn->delta = prev->delta;
        }
        layer_backward(l, bpn);
    }

    /* update phase */
    for(int i = 0; i < bpn->num_layers; i++) {
        layer_update(xcsf, &bpn->layers[i]);
    }
} 

double neural_output(XCSF *xcsf, BPN *bpn, int i)
{
    if(i < bpn->num_outputs) {
        double *output = layer_output(&bpn->layers[bpn->num_layers-1]);
        return fmax(xcsf->MIN_CON, fmin(xcsf->MAX_CON, output[i]));
    }
    printf("neural_output(): requested (%d) in output layer of size (%d)\n",
            i, bpn->num_outputs);
    exit(EXIT_FAILURE);
}

void neural_print(XCSF *xcsf, BPN *bpn, _Bool print_weights)
{
    (void)xcsf;
    for(int i = 0; i < bpn->num_layers; i++) {
        printf("layer (%d) ", i);
        layer_print(&bpn->layers[i], print_weights);
    }
}
