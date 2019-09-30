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

LAYER *neural_layer_connected_init(XCSF *xcsf, int in, int out, int act, u_int32_t opt)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = CONNECTED;
    l->layer_vptr = &layer_connected_vtbl;
    l->function = act;
    l->num_inputs = in;
    l->num_outputs = out;
    l->num_weights = in*out;
    l->state = calloc(l->num_outputs, sizeof(double));
    l->output = calloc(l->num_outputs, sizeof(double));
    l->biases = calloc(l->num_outputs, sizeof(double));
    l->bias_updates = calloc(l->num_outputs, sizeof(double));
    l->weight_updates = calloc(l->num_weights, sizeof(double));
    l->delta = calloc(l->num_outputs, sizeof(double));
    l->weights = malloc(l->num_weights * sizeof(double));
    for(int i = 0; i < l->num_weights; i++) {
        l->weights[i] = rand_normal(0,0.1);
    }
    l->active = calloc(l->num_outputs, sizeof(_Bool));
    l->options = opt;
    if(l->options & LAYER_EVOLVE_NEURONS) {
        l->num_active = 1;// + irand_uniform(0,l->num_outputs);
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
    l->function = from->function;
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
            l->output[i] = neural_activate(l->function, l->state[i]);
        }
        else {
            l->state[i] = 0;
            l->output[i] = 0;
        }
    }
}

void neural_layer_connected_backward(XCSF *xcsf, LAYER *l, NET *net)
{
    // net->input[] = this layer's input
    // net->delta[] = previous layer's delta
    (void)xcsf;
    for(int i = 0; i < l->num_active; i++) {
        l->delta[i] *= neural_gradient(l->function, l->state[i]);
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
    if(l->options & LAYER_SGD_WEIGHTS) {
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
}

_Bool neural_layer_connected_mutate(XCSF *xcsf, LAYER *l)
{
    _Bool mod = false;

    if(l->options & LAYER_EVOLVE_NEURONS) {
        if(rand_uniform(0,1) < xcsf->P_MUTATION) {
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
    } 

    if(l->options & LAYER_EVOLVE_WEIGHTS) {
        int w = l->num_inputs * l->num_active;
        for(int i = 0; i < w; i++) {
            double orig = l->weights[i];
            l->weights[i] += rand_normal(0, xcsf->S_MUTATION);
            if(l->weights[i] != orig) {
                mod = true;
            }
        }
        for(int i = 0; i < l->num_active; i++) {
            double orig = l->biases[i];
            l->biases[i] += rand_normal(0, xcsf->S_MUTATION);
            if(l->biases[i] != orig) {
                mod = true;
            }
        }
    }

    if(l->options & LAYER_EVOLVE_FUNCTIONS) {
        if(rand_uniform(0,1) < xcsf->P_FUNC_MUTATION) {
            l->function = irand_uniform(0,NUM_ACTIVATIONS);
            mod = true;
        } 
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
            activation_string(l->function), l->num_inputs, l->num_outputs, l->num_active);
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

size_t neural_layer_connected_save(XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&l->num_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->num_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->num_weights, sizeof(int), 1, fp);
    s += fwrite(&l->options, sizeof(u_int32_t), 1, fp);
    s += fwrite(&l->num_active, sizeof(int), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(l->active, sizeof(_Bool), l->num_outputs, fp);
    s += fwrite(l->weights, sizeof(double), l->num_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->num_outputs, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->num_outputs, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->num_weights, fp);
    //printf("neural layer connected saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t neural_layer_connected_load(XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&l->num_inputs, sizeof(int), 1, fp);
    s += fread(&l->num_outputs, sizeof(int), 1, fp);
    s += fread(&l->num_weights, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(u_int32_t), 1, fp);
    s += fread(&l->num_active, sizeof(int), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    l->state = calloc(l->num_outputs, sizeof(double));
    l->output = calloc(l->num_outputs, sizeof(double));
    l->delta = calloc(l->num_outputs, sizeof(double));
    l->active = malloc(l->num_outputs * sizeof(_Bool));
    l->weights = malloc(l->num_weights * sizeof(double));
    l->biases = malloc(l->num_outputs * sizeof(double));
    l->bias_updates = malloc(l->num_outputs * sizeof(double));
    l->weight_updates = malloc(l->num_weights * sizeof(double));
    s += fread(l->active, sizeof(_Bool), l->num_outputs, fp);
    s += fread(l->weights, sizeof(double), l->num_weights, fp);
    s += fread(l->biases, sizeof(double), l->num_outputs, fp);
    s += fread(l->bias_updates, sizeof(double), l->num_outputs, fp);
    s += fread(l->weight_updates, sizeof(double), l->num_weights, fp);
    //printf("neural layer connected loaded %lu elements\n", (unsigned long)s);
    return s;
}
