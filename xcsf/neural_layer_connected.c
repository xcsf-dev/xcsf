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
#include "xcsf.h"
#include "utils.h"
#include "blas.h"
#include "sam.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"

#define N_MU 4 //!< Number of mutation rates applied
#define ETA_MAX 0.1 //!< Maximum gradient descent rate
#define ETA_MIN 0.0001 //!< Minimum gradient descent rate

static _Bool mutate_eta(LAYER *l, double mu);
static _Bool mutate_neurons(const XCSF *xcsf, LAYER *l, double mu);
static _Bool mutate_weights(const LAYER *l, double mu);
static _Bool mutate_functions(LAYER *l, double mu);
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
    l->n_weights = in * n_init;
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->biases = calloc(l->n_outputs, sizeof(double));
    l->bias_updates = calloc(l->n_outputs, sizeof(double));
    l->weight_updates = calloc(l->n_weights, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->weights = malloc(l->n_weights * sizeof(double));
    for(int i = 0; i < l->n_weights; i++) {
        l->weights[i] = rand_normal(0,0.1);
    }
    l->options = o;
    if(l->options & LAYER_EVOLVE_ETA) {
        l->eta = rand_uniform(ETA_MIN,ETA_MAX);
    }
    else {
        l->eta = xcsf->PRED_ETA;
    }
    l->mu = malloc(N_MU * sizeof(double));
    sam_init(xcsf, l->mu, N_MU);
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
    l->n_weights = from->n_weights;
    l->state = calloc(from->n_outputs, sizeof(double));
    l->output = calloc(from->n_outputs, sizeof(double));
    l->biases = malloc(from->n_outputs * sizeof(double));
    l->bias_updates = calloc(from->n_outputs, sizeof(double));
    l->weight_updates = calloc(from->n_weights, sizeof(double));
    l->delta = calloc(from->n_outputs, sizeof(double));
    l->weights = malloc(from->n_weights * sizeof(double));
    memcpy(l->weights, from->weights, from->n_weights * sizeof(double));
    memcpy(l->biases, from->biases, from->n_outputs * sizeof(double));
    l->options = from->options;
    l->eta = from->eta;
    l->mu = malloc(N_MU * sizeof(double));
    memcpy(l->mu, from->mu, N_MU * sizeof(double));
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
    free(l->mu);
}

void neural_layer_connected_rand(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    for(int i = 0; i < l->n_weights; i++) {
        l->weights[i] = rand_normal(0,1);
    }
    for(int i = 0; i < l->n_outputs; i++) {
        l->biases[i] = rand_normal(0,1);
    }
}

void neural_layer_connected_forward(const XCSF *xcsf, const LAYER *l, const double *input)
{
    (void)xcsf;
    int k = l->n_inputs;
    int n = l->n_outputs;
    const double *a = input;
    double *b = l->weights;
    double *c = l->state;
    // states = biases
    memcpy(l->state, l->biases, sizeof(double) * l->n_outputs);
    // states += weights * inputs
    blas_gemm(0,1,1,n,k,1,a,k,b,k,1,c,n);
    // apply activations
    for(int i = 0; i < l->n_outputs; i++) {
        l->state[i] = constrain(-100, 100, l->state[i]);
        l->output[i] = neural_activate(l->function, l->state[i]);
    }
}

void neural_layer_connected_backward(const XCSF *xcsf, const LAYER *l, const NET *net)
{
    // net->input[] = this layer's input
    // net->delta[] = previous layer's delta
    (void)xcsf;
    // apply gradients
    for(int i = 0; i < l->n_outputs; i++) {
        l->delta[i] *= neural_gradient(l->function, l->state[i]);
    }
    // calculate updates
    if(l->options & LAYER_SGD_WEIGHTS) {
        int m = l->n_outputs;
        int n = l->n_inputs;
        const double *a = l->delta;
        const double *b = net->input;
        double *c = l->weight_updates;
        blas_axpy(l->n_outputs, 1, l->delta, 1, l->bias_updates, 1);
        blas_gemm(1,0,m,n,1,1,a,m,b,n,1,c,n);
    }
    // set the error for the previous layer (if there is one)
    if(net->delta) {
        int k = l->n_outputs;
        int n = l->n_inputs;
        const double *a = l->delta;
        const double *b = l->weights;
        double *c = net->delta;
        blas_gemm(0,0,1,n,k,1,a,k,b,n,1,c,n);
    }
}

void neural_layer_connected_update(const XCSF *xcsf, const LAYER *l)
{
    if(l->options & LAYER_SGD_WEIGHTS) {
        blas_axpy(l->n_outputs, l->eta, l->bias_updates, 1, l->biases, 1);
        blas_scal(l->n_outputs, xcsf->PRED_MOMENTUM, l->bias_updates, 1);
        blas_axpy(l->n_weights, l->eta, l->weight_updates, 1, l->weights, 1);
        blas_scal(l->n_weights, xcsf->PRED_MOMENTUM, l->weight_updates, 1);
    }
}

void neural_layer_connected_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void)xcsf;
    int n_weights = prev->n_outputs * l->n_outputs;
    double *weights = malloc(n_weights * sizeof(double));
    double *weight_updates = malloc(n_weights * sizeof(double));
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
    l->n_weights = n_weights;
    l->n_inputs = prev->n_outputs;
}

_Bool neural_layer_connected_mutate(const XCSF *xcsf, LAYER *l)
{
    sam_adapt(xcsf, l->mu, N_MU);
    _Bool mod = false;
    if((l->options & LAYER_EVOLVE_ETA) && mutate_eta(l, l->mu[0])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_NEURONS) && mutate_neurons(xcsf, l, l->mu[1])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_WEIGHTS) && mutate_weights(l, l->mu[2])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_FUNCTIONS) && mutate_functions(l, l->mu[3])) {
        mod = true;
    }
    return mod;
}

static _Bool mutate_eta(LAYER *l, double mu)
{
    double orig = l->eta;
    l->eta += rand_normal(0, mu);
    l->eta = constrain(ETA_MIN, ETA_MAX, l->eta);
    if(l->eta != orig) {
        return true;
    }
    return false;
}

static _Bool mutate_neurons(const XCSF *xcsf, LAYER *l, double mu)
{
    _Bool mod = false;
    if(rand_uniform(0,1) < mu) {
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
    int n_weights = l->n_outputs * l->n_inputs;
    double *weights = malloc(n_weights * sizeof(double));
    double *weight_updates = malloc(n_weights * sizeof(double));
    double *state = calloc(l->n_outputs, sizeof(double));
    double *output = calloc(l->n_outputs, sizeof(double));
    double *biases = malloc(l->n_outputs * sizeof(double));
    double *bias_updates = malloc(l->n_outputs * sizeof(double));
    double *delta = calloc(l->n_outputs, sizeof(double));
    memcpy(weights, l->weights, l->n_weights * sizeof(double));
    memcpy(weight_updates, l->weight_updates, l->n_weights * sizeof(double));
    for(int i = l->n_weights; i < n_weights; i++) {
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
    l->n_weights = n_weights;
}

static void neuron_remove(LAYER *l, int n)
{
    l->n_outputs -= n;
    int n_weights = l->n_outputs * l->n_inputs;
    double *weights = malloc(n_weights * sizeof(double));
    double *weight_updates = malloc(n_weights * sizeof(double));
    double *state = calloc(l->n_outputs, sizeof(double));
    double *output = calloc(l->n_outputs, sizeof(double));
    double *biases = malloc(l->n_outputs * sizeof(double));
    double *bias_updates = malloc(l->n_outputs * sizeof(double));
    double *delta = calloc(l->n_outputs, sizeof(double));
    memcpy(weights, l->weights, n_weights * sizeof(double));
    memcpy(weight_updates, l->weight_updates, n_weights * sizeof(double));
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
    l->n_weights = n_weights;
}

static _Bool mutate_weights(const LAYER *l, double mu)
{
    _Bool mod = false;
    for(int i = 0; i < l->n_weights; i++) {
        double orig = l->weights[i];
        l->weights[i] += rand_normal(0, mu);
        if(l->weights[i] != orig) {
            mod = true;
        }
    }
    for(int i = 0; i < l->n_outputs; i++) {
        double orig = l->biases[i];
        l->biases[i] += rand_normal(0, mu);
        if(l->biases[i] != orig) {
            mod = true;
        }
    }
    return mod;
}

static _Bool mutate_functions(LAYER *l, double mu)
{
    if(rand_uniform(0,1) < mu) {
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
    printf("connected %s, in = %d, out = %d, ",
            activation_string(l->function), l->n_inputs, l->n_outputs);
    printf("weights (%d): ", l->n_weights);
    if(print_weights) {
        for(int i = 0; i < l->n_weights; i++) {
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
    s += fwrite(&l->n_weights, sizeof(int), 1, fp);
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(l->weights, sizeof(double), l->n_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t neural_layer_connected_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_weights, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->weights = malloc(l->n_weights * sizeof(double));
    l->biases = malloc(l->n_outputs * sizeof(double));
    l->bias_updates = malloc(l->n_outputs * sizeof(double));
    l->weight_updates = malloc(l->n_weights * sizeof(double));
    s += fread(l->weights, sizeof(double), l->n_weights, fp);
    s += fread(l->biases, sizeof(double), l->n_outputs, fp);
    s += fread(l->bias_updates, sizeof(double), l->n_outputs, fp);
    s += fread(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    return s;
}
