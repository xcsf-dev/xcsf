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

LAYER *neural_layer_connected_init(XCSF *xcsf, int in, int out, int act, int opt)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = CONNECTED;
    l->layer_vptr = &layer_connected_vtbl;
    l->activation_type = act;
    activation_set(&l->activate, act);
    gradient_set(&l->gradient, act);
    l->num_inputs = in;
    l->num_outputs = out;
    l->num_weights = in*out;
    l->state = calloc(l->num_outputs, sizeof(double));
    l->output = calloc(l->num_outputs, sizeof(double));
    l->biases = calloc(l->num_outputs, sizeof(double));
    l->bias_updates = calloc(l->num_outputs, sizeof(double));
    l->weight_updates = calloc(l->num_weights, sizeof(double));
    l->delta = calloc(l->num_outputs, sizeof(double));
    // small random weights
    l->weights = malloc(l->num_weights * sizeof(double));
    for(int i = 0; i < l->num_weights; i++) {
        l->weights[i] = rand_normal(0,0.1);
    }
    // initial number of active neurons
    l->active = calloc(l->num_outputs, sizeof(_Bool));
    l->options = opt;
    if(l->options > 0) {
        l->num_active = 1;
    }
    else {
        l->num_active = l->num_outputs;
    }
    for(int i = 0; i < l->num_active; i++) {
        l->active[i] = true;
    }
    return l;
}

LAYER *neural_layer_connected_copy(XCSF *xcsf, LAYER *from)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = from->layer_type;
    l->layer_vptr = from->layer_vptr;
    l->num_inputs = from->num_inputs;
    l->num_outputs = from->num_outputs;
    l->num_weights = from->num_weights;
    l->options = from->options;
    l->num_active = from->num_active;
    l->active = malloc(from->num_outputs * sizeof(_Bool));
    l->state = calloc(from->num_outputs, sizeof(double));
    l->output = calloc(from->num_outputs, sizeof(double));
    l->weights = malloc(from->num_weights * sizeof(double));
    l->biases = malloc(from->num_outputs * sizeof(double));
    l->bias_updates = calloc(from->num_outputs, sizeof(double));
    l->weight_updates = calloc(from->num_weights, sizeof(double));
    l->delta = calloc(from->num_outputs, sizeof(double));
    l->activation_type = from->activation_type;
    activation_set(&l->activate, from->activation_type);
    gradient_set(&l->gradient, from->activation_type);
    memcpy(l->weights, from->weights, from->num_weights * sizeof(double));
    memcpy(l->biases, from->biases, from->num_outputs * sizeof(double));
    memcpy(l->active, from->active, from->num_outputs * sizeof(_Bool));
    return l;
}

void neural_layer_connected_free(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    free(l->active);
    free(l->state);
    free(l->output);
    free(l->weights);
    free(l->biases);
    free(l->bias_updates);
    free(l->weight_updates);
    free(l->delta);
}

void neural_layer_connected_rand(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    for(int i = 0; i < l->num_weights; i++) {
        l->weights[i] = rand_normal(0,1);
    }
    for(int i = 0; i < l->num_outputs; i++) {
        l->biases[i] = rand_normal(0,1);
    }
}

void neural_layer_connected_forward(XCSF *xcsf, LAYER *l, double *input)
{
    (void)xcsf;
    for(int i = 0; i < l->num_outputs; i++) {
        if(l->active[i]) {
            l->state[i] = 0;
            for(int j = 0; j < l->num_inputs; j++) {
                l->state[i] += input[j] * l->weights[i*l->num_inputs+j];
            }
            l->state[i] += l->biases[i];
            l->state[i] = constrain(-100, 100, l->state[i]);
            l->output[i] = (l->activate)(l->state[i]);
        }
        else {
            l->state[i] = 0;
            l->output[i] = 0;
        }
    }
}

void neural_layer_connected_backward(XCSF *xcsf, LAYER *l, NET *net)
{
    (void)xcsf;
    // net input = this layer's input
    // net delta = previous layer's delta
    for(int i = 0; i < l->num_active; i++) {
        l->delta[i] *= (l->gradient)(l->state[i]);
        l->bias_updates[i] += l->delta[i];
        for(int j = 0; j < l->num_inputs; j++) {
            l->weight_updates[i*l->num_inputs+j] += l->delta[i] * net->input[j];
        }
    }   
    if(net->delta) { // input layer has no delta or weights
        for(int i = 0; i < l->num_active; i++) {
            for(int j = 0; j < l->num_inputs; j++) {
                net->delta[j] += l->delta[i] * l->weights[i*l->num_inputs+j];
            }
        }
    }
}

void neural_layer_connected_update(XCSF *xcsf, LAYER *l)
{
    for(int i = 0; i < l->num_active; i++) {
        l->biases[i] += xcsf->ETA * l->bias_updates[i];
        l->bias_updates[i] *= xcsf->MOMENTUM;
    }
    int w = l->num_inputs * l->num_active;
    for(int i = 0; i < w; i++) {
        l->weights[i] += xcsf->ETA * l->weight_updates[i];
        l->weight_updates[i] *= xcsf->MOMENTUM;
    }
}

_Bool neural_layer_connected_mutate(XCSF *xcsf, LAYER *l)
{
    _Bool mod = false;
    // mutate number of neurons
    if(l->options > 0 && rand_uniform(0,1) < xcsf->P_MUTATION) {
        int idx = l->num_active - 1;
        // remove
        if(l->num_active > 1 && rand_uniform(0,1) < 0.5) {
            l->active[idx] = false;
            l->num_active--;
            mod = true;
        }
        // add
        else if(l->num_active < l->num_outputs) {
            l->active[idx] = true;
            l->num_active++;
            // randomise weights
            l->biases[idx] = 0;
            for(int i = 0; i < l->num_inputs; i++) {
                l->weights[idx*l->num_inputs+i] = rand_normal(0,0.1);
            }
            mod = true;
        }
    } 
    // mutate weights
    int w = l->num_inputs * l->num_active;
    for(int i = 0; i < w; i++) {
        double orig = l->weights[i];
        l->weights[i] += rand_normal(0, xcsf->S_MUTATION);
        if(l->weights[i] != orig) {
            mod = true;
        }
    }
    // mutate biases
    for(int i = 0; i < l->num_active; i++) {
        double orig = l->biases[i];
        l->biases[i] += rand_normal(0, xcsf->S_MUTATION);
        if(l->biases[i] != orig) {
            mod = true;
        }
    }
    // mutate activation functions
    if(rand_uniform(0,1) < xcsf->P_FUNC_MUTATION) {
        l->activation_type = irand_uniform(0,NUM_ACTIVATIONS);
        activation_set(&l->activate, l->activation_type);
        gradient_set(&l->gradient, l->activation_type);
        mod = true;
    } 
    return mod;
}

double *neural_layer_connected_output(XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    return l->output;
}

void neural_layer_connected_print(XCSF *xcsf, LAYER *l, _Bool print_weights)
{
    (void)xcsf;
    printf("connected %s in = %d, out = %d, active = %d, ",
            activation_string(l->activation_type), l->num_inputs, l->num_outputs, l->num_active);
    printf("weights (%d): ", l->num_weights);
    if(print_weights) {
        for(int i = 0; i < l->num_weights; i++) {
            printf("%.4f, ", l->weights[i]);
        }
    }
    printf("biases (%d): ", l->num_outputs);
    if(print_weights) {
        for(int i = 0; i < l->num_outputs; i++) {
            printf("%.4f, ", l->biases[i]);
        }
    }
    printf("\n");
}
