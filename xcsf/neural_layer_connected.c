/*
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
 */
       
/**
 * @file neural_layer_connected.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a fully-connected layer of perceptrons.
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"

#define ETA_MAX 0.1 //!< maximum gradient descent rate
#define ETA_MIN 0.0001 //!< minimum gradient descent rate 

static _Bool mutate_neurons(const XCSF *xcsf, LAYER *l);
static _Bool mutate_weights(const XCSF *xcsf, const LAYER *l);
static _Bool mutate_eta(const XCSF *xcsf, LAYER *l);
static _Bool mutate_functions(const XCSF *xcsf, LAYER *l);
static void neuron_add(LAYER *l, int n);
static void neuron_remove(LAYER *l, int n);

LAYER *neural_layer_connected_init(const XCSF *xcsf, int in, int n_init, int n_max, int f, uint32_t o)
{
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = CONNECTED;
    l->layer_vptr = &layer_connected_vtbl;
    l->function = f;
    l->n_inputs = in;
    l->n_outputs = n_init;
    l->max_outputs = n_max;
    l->num_weights = in * n_init;
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->biases = calloc(l->n_outputs, sizeof(double));
    l->bias_updates = calloc(l->n_outputs, sizeof(double));
    l->weight_updates = calloc(l->num_weights, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->weights = malloc(l->num_weights * sizeof(double));
    for(int i = 0; i < l->num_weights; i++) {
        l->weights[i] = rand_normal(0,0.1);
    }
    l->options = o;
    if(l->options & LAYER_EVOLVE_ETA) {
        l->eta = rand_uniform(ETA_MIN,ETA_MAX);
    }
    else {
        l->eta = xcsf->PRED_ETA;
    }
    return l;
}

LAYER *neural_layer_connected_copy(const XCSF *xcsf, const LAYER *from)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = from->layer_type;
    l->layer_vptr = from->layer_vptr;
    l->function = from->function;
    l->n_inputs = from->n_inputs;
    l->n_outputs = from->n_outputs;
    l->max_outputs = from->max_outputs;
    l->num_weights = from->num_weights;
    l->state = calloc(from->n_outputs, sizeof(double));
    l->output = calloc(from->n_outputs, sizeof(double));
    l->biases = malloc(from->n_outputs * sizeof(double));
    l->bias_updates = calloc(from->n_outputs, sizeof(double));
    l->weight_updates = calloc(from->num_weights, sizeof(double));
    l->delta = calloc(from->n_outputs, sizeof(double));
    l->weights = malloc(from->num_weights * sizeof(double));
    memcpy(l->weights, from->weights, from->num_weights * sizeof(double));
    memcpy(l->biases, from->biases, from->n_outputs * sizeof(double));
    l->options = from->options;
    l->eta = from->eta;
    return l;
}

void neural_layer_connected_free(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    free(l->state);
    free(l->output);
    free(l->weights);
    free(l->biases);
    free(l->bias_updates);
    free(l->weight_updates);
    free(l->delta);
}

void neural_layer_connected_rand(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    for(int i = 0; i < l->num_weights; i++) {
        l->weights[i] = rand_normal(0,1);
    }
    for(int i = 0; i < l->n_outputs; i++) {
        l->biases[i] = rand_normal(0,1);
    }
}

void neural_layer_connected_forward(const XCSF *xcsf, const LAYER *l, const double *input)
{
    (void)xcsf;
    for(int i = 0; i < l->n_outputs; i++) {
        l->state[i] = l->biases[i];
        int offset = i * l->n_inputs;
        for(int j = 0; j < l->n_inputs; j++) {
            l->state[i] += input[j] * l->weights[offset + j];
        }
        l->state[i] = constrain(-100, 100, l->state[i]);
        l->output[i] = neural_activate(l->function, l->state[i]);
    }
}

void neural_layer_connected_backward(const XCSF *xcsf, const LAYER *l, const NET *net)
{
    // net->input[] = this layer's input
    // net->delta[] = previous layer's delta
    (void)xcsf;
    for(int i = 0; i < l->n_outputs; i++) {
        l->delta[i] *= neural_gradient(l->function, l->state[i]);
        if(l->options & LAYER_SGD_WEIGHTS) {
            l->bias_updates[i] += l->delta[i];
            int offset = i * l->n_inputs;
            for(int j = 0; j < l->n_inputs; j++) {
                l->weight_updates[offset + j] += l->delta[i] * net->input[j];
            }
        }
    }
    if(net->delta) { // input layer has no delta or weights
        for(int i = 0; i < l->n_outputs; i++) {
            int offset = i * l->n_inputs;
            for(int j = 0; j < l->n_inputs; j++) {
                net->delta[j] += l->delta[i] * l->weights[offset + j];
            }
        }
    }
}

void neural_layer_connected_update(const XCSF *xcsf, const LAYER *l)
{
    if(l->options & LAYER_SGD_WEIGHTS) {
        for(int i = 0; i < l->n_outputs; i++) {
            l->biases[i] += l->eta * l->bias_updates[i];
            l->bias_updates[i] *= xcsf->PRED_MOMENTUM;
        }
        for(int i = 0; i < l->num_weights; i++) {
            l->weights[i] += l->eta * l->weight_updates[i];
            l->weight_updates[i] *= xcsf->PRED_MOMENTUM;
        }
    }
}

void neural_layer_connected_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void)xcsf;
    int num_weights = prev->n_outputs * l->n_outputs;
    double *weights = malloc(num_weights * sizeof(double));
    double *weight_updates = malloc(num_weights * sizeof(double));
    for(int i = 0; i < l->n_outputs; i++) {
        int orig_offset = i * l->n_inputs;
        int offset = i * prev->n_outputs;
        for(int j = 0; j < prev->n_outputs; j++) {
            if(j < l->n_inputs) {
                weights[offset + j] = l->weights[orig_offset + j];
                weight_updates[offset + j] = l->weight_updates[orig_offset + j];
            }
            else {
                weights[offset + j] = rand_normal(0,0.1);
                weight_updates[offset + j] = 0;
            }
        }
    }
    free(l->weights);
    free(l->weight_updates);
    l->weights = weights;
    l->weight_updates = weight_updates;
    l->num_weights = num_weights;
    l->n_inputs = prev->n_outputs;
}

_Bool neural_layer_connected_mutate(const XCSF *xcsf, LAYER *l)
{
    _Bool mod = false;
    if((l->options & LAYER_EVOLVE_NEURONS) && mutate_neurons(xcsf, l)) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_WEIGHTS) && mutate_weights(xcsf, l)) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_ETA) && mutate_eta(xcsf, l)) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_FUNCTIONS) && mutate_functions(xcsf, l)) {
        mod = true;
    }
    return mod;
}

static _Bool mutate_neurons(const XCSF *xcsf, LAYER *l)
{
    _Bool mod = false;
    if(rand_uniform(0,1) < xcsf->P_MUTATION) {
        // number of neurons to add or remove
        int n = 1 + irand_uniform(0, xcsf->MAX_NEURON_MOD);
        // remove neurons
        if(rand_uniform(0,1) < 0.5) {
            if(l->n_outputs - n < 1) {
                n = l->n_outputs - 1;
            }
            if(n > 0) {
                neuron_remove(l, n);
                mod = true;
            }
        }
        // add neurons
        else {
            if(l->n_outputs + n > l->max_outputs) {
                n = l->max_outputs - l->n_outputs;
            }
            if(n > 0) {
                neuron_add(l, n);
                mod = true;
            }
        }
    }
    return mod;
}

static void neuron_add(LAYER *l, int n)
{
    l->n_outputs += n;
    int num_weights = l->n_outputs * l->n_inputs;
    double *weights = malloc(num_weights * sizeof(double));
    double *weight_updates = malloc(num_weights * sizeof(double));
    double *state = calloc(l->n_outputs, sizeof(double));
    double *output = calloc(l->n_outputs, sizeof(double));
    double *biases = malloc(l->n_outputs * sizeof(double));
    double *bias_updates = malloc(l->n_outputs * sizeof(double));
    double *delta = calloc(l->n_outputs, sizeof(double));
    memcpy(weights, l->weights, l->num_weights * sizeof(double));
    memcpy(weight_updates, l->weight_updates, l->num_weights * sizeof(double));
    for(int i = l->num_weights; i < num_weights; i++) {
        weights[i] = rand_normal(0,0.1);
        weight_updates[i] = 0;
    }
    memcpy(biases, l->biases, (l->n_outputs - n) * sizeof(double));
    memcpy(bias_updates, l->bias_updates, (l->n_outputs - n) * sizeof(double));
    for(int i = l->n_outputs - n; i < l->n_outputs; i++) {
        biases[i] = 0;
        bias_updates[i] = 0;
    }
    free(l->weights);
    free(l->weight_updates);
    free(l->state);
    free(l->output);
    free(l->biases);
    free(l->bias_updates);
    free(l->delta);
    l->weights = weights;
    l->weight_updates = weight_updates;
    l->state = state;
    l->output = output;
    l->biases = biases;
    l->bias_updates = bias_updates;
    l->delta = delta;
    l->num_weights = num_weights;
}

static void neuron_remove(LAYER *l, int n)
{
    l->n_outputs -= n;
    int num_weights = l->n_outputs * l->n_inputs;
    double *weights = malloc(num_weights * sizeof(double));
    double *weight_updates = malloc(num_weights * sizeof(double));
    double *state = calloc(l->n_outputs, sizeof(double));
    double *output = calloc(l->n_outputs, sizeof(double));
    double *biases = malloc(l->n_outputs * sizeof(double));
    double *bias_updates = malloc(l->n_outputs * sizeof(double));
    double *delta = calloc(l->n_outputs, sizeof(double));
    memcpy(weights, l->weights, num_weights * sizeof(double));
    memcpy(weight_updates, l->weight_updates, num_weights * sizeof(double));
    memcpy(biases, l->biases, l->n_outputs * sizeof(double));
    memcpy(bias_updates, l->bias_updates, l->n_outputs * sizeof(double));
    free(l->weights);
    free(l->weight_updates);
    free(l->state);
    free(l->output);
    free(l->biases);
    free(l->bias_updates);
    free(l->delta);
    l->weights = weights;
    l->weight_updates = weight_updates;
    l->state = state;
    l->output = output;
    l->biases = biases;
    l->bias_updates = bias_updates;
    l->delta = delta;
    l->num_weights = num_weights;
}

static _Bool mutate_weights(const XCSF *xcsf, const LAYER *l)
{
    _Bool mod = false;
    for(int i = 0; i < l->num_weights; i++) {
        double orig = l->weights[i];
        l->weights[i] += rand_normal(0, xcsf->S_MUTATION);
        if(l->weights[i] != orig) {
            mod = true;
        }
    }
    for(int i = 0; i < l->n_outputs; i++) {
        double orig = l->biases[i];
        l->biases[i] += rand_normal(0, xcsf->S_MUTATION);
        if(l->biases[i] != orig) {
            mod = true;
        }
    }
    return mod;
}

static _Bool mutate_eta(const XCSF *xcsf, LAYER *l)
{
    double orig = l->eta;
    l->eta += rand_normal(0, xcsf->E_MUTATION);
    l->eta = constrain(ETA_MIN, ETA_MAX, l->eta);
    if(l->eta != orig) {
        return true;
    }
    return false;
}

static _Bool mutate_functions(const XCSF *xcsf, LAYER *l)
{
    if(rand_uniform(0,1) < xcsf->F_MUTATION) {
        int orig = l->function;
        l->function = irand_uniform(0, NUM_ACTIVATIONS);
        if(l->function != orig) {
            return true;
        }
    }
    return false;
}

double *neural_layer_connected_output(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    return l->output;
}

void neural_layer_connected_print(const XCSF *xcsf, const LAYER *l, _Bool print_weights)
{
    (void)xcsf;
    printf("connected %s eta=%.5f, in = %d, out = %d, ",
            activation_string(l->function), l->eta, l->n_inputs, l->n_outputs);
    printf("weights (%d): ", l->num_weights);
    if(print_weights) {
        for(int i = 0; i < l->num_weights; i++) {
            printf("%.4f, ", l->weights[i]);
        }
    }
    printf("biases (%d): ", l->n_outputs);
    if(print_weights) {
        for(int i = 0; i < l->n_outputs; i++) {
            printf("%.4f, ", l->biases[i]);
        }
    }
    printf("\n");
}

size_t neural_layer_connected_save(const XCSF *xcsf, const LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->num_weights, sizeof(int), 1, fp);
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(l->weights, sizeof(double), l->num_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->num_weights, fp);
    return s;
}

size_t neural_layer_connected_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->num_weights, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->weights = malloc(l->num_weights * sizeof(double));
    l->biases = malloc(l->n_outputs * sizeof(double));
    l->bias_updates = malloc(l->n_outputs * sizeof(double));
    l->weight_updates = malloc(l->num_weights * sizeof(double));
    s += fread(l->weights, sizeof(double), l->num_weights, fp);
    s += fread(l->biases, sizeof(double), l->n_outputs, fp);
    s += fread(l->bias_updates, sizeof(double), l->n_outputs, fp);
    s += fread(l->weight_updates, sizeof(double), l->num_weights, fp);
    return s;
}
