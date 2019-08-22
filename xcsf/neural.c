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
#include "random.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer_connected.h"

void neural_init(XCSF *xcsf, BPN *bpn, int num_layers, int *neurons, int *activations)
{
    (void)xcsf;
    bpn->num_layers = num_layers-1; // number of hidden and output layers
    bpn->num_inputs = neurons[0];
    bpn->num_outputs = neurons[num_layers-1];
    bpn->layers = malloc(bpn->num_layers*sizeof(LAYER));
    for(int i = 0; i < bpn->num_layers; i++) {
        neural_layer_connected_init(&bpn->layers[i], neurons[i], neurons[i+1], activations[i]);
    }
}

void neural_copy(XCSF *xcsf, BPN *to, BPN *from)
{
    (void)xcsf;
    to->num_layers = from->num_layers;
    to->num_outputs = from->num_outputs;
    to->num_inputs = from->num_inputs;
    for(int i = 0; i < from->num_layers; i++) {
        neural_layer_connected_copy(&to->layers[i], &from->layers[i]);
    }
}

void neural_free(XCSF *xcsf, BPN *bpn)
{
    (void)xcsf;
    for(int i = 0; i < bpn->num_layers; i++) {
        neural_layer_connected_free(&bpn->layers[i]);
    }
}

void neural_rand(XCSF *xcsf, BPN *bpn)
{
    (void)xcsf;
    for(int i = 0; i < bpn->num_layers; i++) {
        neural_layer_connected_rand(&bpn->layers[i]);
    }
}    

_Bool neural_mutate(XCSF *xcsf, BPN *bpn)
{
    _Bool mod = false;
    for(int i = 0; i < bpn->num_layers; i++) {
        if(neural_layer_connected_mutate(xcsf, &bpn->layers[i])) {
            mod = true;
        }
    }
    return mod;
}
 

void neural_propagate(XCSF *xcsf, BPN *bpn, double *input)
{
    (void)xcsf;
    neural_layer_connected_forward(&bpn->layers[0], input);
    for(int i = 1; i < bpn->num_layers; i++) {
        neural_layer_connected_forward(&bpn->layers[i], bpn->layers[i-1].output);
    }
}

double neural_output(XCSF *xcsf, BPN *bpn, int i)
{
    if(i < bpn->num_outputs) {
        double out = bpn->layers[bpn->num_layers-1].output[i];
        return fmax(xcsf->MIN_CON, fmin(xcsf->MAX_CON, out));
    }
    printf("error: requested output (%d) in output layer of size (%d)\n",
        i, bpn->num_outputs);
    exit(EXIT_FAILURE);
}    

void neural_print(XCSF *xcsf, BPN *bpn, _Bool print_weights)
{
    (void)xcsf;
    for(int i = 0; i < bpn->num_layers; i++) {
        printf("layer (%d) ", i);
        neural_layer_connected_print(&bpn->layers[i], print_weights);
    }
}        

void neural_learn(XCSF *xcsf, BPN *bpn, double *truth, double *input)
{
    (void)input; // input already propagated in set_pred()
    /* backward phase */
    // output layer
    LAYER *p = &bpn->layers[bpn->num_layers-1];
    for(int i = 0; i < p->num_outputs; i++) {
        p->delta[i] = (truth[i] - p->output[i]);
    }
    neural_layer_connected_backward(p);
    // hidden layers
    for(int i = bpn->num_layers-2; i >= 0; i--) {
        LAYER *l = &bpn->layers[i];
        for(int j = 0; j < l->num_outputs; j++) {
            // this layer uses the next layer's error
            l->delta[j] = 0.0;
            for(int k = 0; k < p->num_outputs; k++) {
                l->delta[j] += p->delta[k] * p->weights[k*p->num_inputs+j];
            }
        }
        neural_layer_connected_backward(l);
        p = l;
    }
    /* update phase */
    for(int i = 0; i < bpn->num_layers; i++) {
        neural_layer_connected_update(xcsf, &bpn->layers[i]);
    }
}
