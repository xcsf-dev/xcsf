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
#include <limits.h>
#include "xcsf.h"
#include "utils.h"
#include "blas.h"
#include "sam.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"

#define N_MU 5 //!< Number of mutation rates applied to a connected layer

LAYER *neural_layer_connected_init(const XCSF *xcsf, int in, int n_init, int n_max, int f,
                                   uint32_t o)
{
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = CONNECTED;
    l->layer_vptr = &layer_connected_vtbl;
    l->options = o;
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
    l->weight_active = malloc(l->n_weights * sizeof(_Bool));
    l->weights = malloc(l->n_weights * sizeof(double));
    l->n_active = l->n_weights;
    for(int i = 0; i < l->n_weights; i++) {
        l->weights[i] = rand_normal(0, 0.1);
        l->weight_active[i] = true;
    }
    if(l->options & LAYER_EVOLVE_ETA) {
        neural_layer_init_eta(xcsf, l);
    } else {
        l->eta = xcsf->PRED_ETA;
    }
    l->mu = malloc(N_MU * sizeof(double));
    sam_init(xcsf, l->mu, N_MU);
    return l;
}

LAYER *neural_layer_connected_copy(const XCSF *xcsf, const LAYER *src)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->function = src->function;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->n_weights = src->n_weights;
    l->state = calloc(src->n_outputs, sizeof(double));
    l->output = calloc(src->n_outputs, sizeof(double));
    l->biases = malloc(src->n_outputs * sizeof(double));
    memcpy(l->biases, src->biases, src->n_outputs * sizeof(double));
    l->bias_updates = calloc(src->n_outputs, sizeof(double));
    l->weight_updates = calloc(src->n_weights, sizeof(double));
    l->delta = calloc(src->n_outputs, sizeof(double));
    l->weights = malloc(src->n_weights * sizeof(double));
    memcpy(l->weights, src->weights, src->n_weights * sizeof(double));
    l->weight_active = malloc(src->n_weights * sizeof(_Bool));
    memcpy(l->weight_active, src->weight_active, src->n_weights * sizeof(_Bool));
    l->options = src->options;
    l->eta = src->eta;
    l->n_active = src->n_active;
    l->mu = malloc(N_MU * sizeof(double));
    memcpy(l->mu, src->mu, N_MU * sizeof(double));
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
    free(l->weight_active);
    free(l->mu);
}

void neural_layer_connected_rand(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    l->n_active = l->n_weights;
    for(int i = 0; i < l->n_weights; i++) {
        l->weights[i] = rand_normal(0, 1);
        l->weight_active[i] = true;
    }
    for(int i = 0; i < l->n_outputs; i++) {
        l->biases[i] = rand_normal(0, 1);
    }
}

void neural_layer_connected_forward(const XCSF *xcsf, const LAYER *l, NET *net)
{
    (void)xcsf;
    int k = l->n_inputs;
    int n = l->n_outputs;
    const double *a = net->input;
    const double *b = l->weights;
    double *c = l->state;
    // states = biases
    memcpy(l->state, l->biases, l->n_outputs * sizeof(double));
    // states += weights * inputs
    blas_gemm(0, 1, 1, n, k, 1, a, k, b, k, 1, c, n);
    // apply activations
    neural_activate_array(l->state, l->output, l->n_outputs, l->function);
}

void neural_layer_connected_backward(const XCSF *xcsf, const LAYER *l, NET *net)
{
    // net->input[] = this layer's input
    // net->delta[] = previous layer's delta
    (void)xcsf;
    // apply gradients
    neural_gradient_array(l->state, l->delta, l->n_outputs, l->function);
    // calculate updates
    if(l->options & LAYER_SGD_WEIGHTS) {
        int m = l->n_outputs;
        int n = l->n_inputs;
        const double *a = l->delta;
        const double *b = net->input;
        double *c = l->weight_updates;
        blas_axpy(l->n_outputs, 1, l->delta, 1, l->bias_updates, 1);
        blas_gemm(1, 0, m, n, 1, 1, a, m, b, n, 1, c, n);
    }
    // set the error for the previous layer (if there is one)
    if(net->delta) {
        int k = l->n_outputs;
        int n = l->n_inputs;
        const double *a = l->delta;
        const double *b = l->weights;
        double *c = net->delta;
        blas_gemm(0, 0, 1, n, k, 1, a, k, b, n, 1, c, n);
    }
}

void neural_layer_connected_update(const XCSF *xcsf, const LAYER *l)
{
    if(l->options & LAYER_SGD_WEIGHTS) {
        blas_axpy(l->n_outputs, l->eta, l->bias_updates, 1, l->biases, 1);
        blas_axpy(l->n_weights, l->eta, l->weight_updates, 1, l->weights, 1);
        blas_scal(l->n_outputs, xcsf->PRED_MOMENTUM, l->bias_updates, 1);
        blas_scal(l->n_weights, xcsf->PRED_MOMENTUM, l->weight_updates, 1);
        neural_layer_weight_clamp(l);
    }
}

void neural_layer_connected_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void)xcsf;
    int n_weights = prev->n_outputs * l->n_outputs;
    double *weights = malloc(n_weights * sizeof(double));
    double *weight_updates = malloc(n_weights * sizeof(double));
    _Bool *weight_active = malloc(n_weights * sizeof(_Bool));
    for(int i = 0; i < l->n_outputs; i++) {
        int orig_offset = i * l->n_inputs;
        int offset = i * prev->n_outputs;
        for(int j = 0; j < prev->n_outputs; j++) {
            if(j < l->n_inputs) {
                weights[offset + j] = l->weights[orig_offset + j];
                weight_updates[offset + j] = l->weight_updates[orig_offset + j];
                weight_active[offset + j] = l->weight_active[orig_offset + j];
            } else {
                weights[offset + j] = rand_normal(0, 0.1);
                weight_updates[offset + j] = 0;
                weight_active[offset + j] = true;
            }
        }
    }
    free(l->weights);
    free(l->weight_updates);
    free(l->weight_active);
    l->weights = weights;
    l->weight_updates = weight_updates;
    l->weight_active = weight_active;
    l->n_weights = n_weights;
    l->n_inputs = prev->n_outputs;
    neural_layer_calc_n_active(l);
}

_Bool neural_layer_connected_mutate(const XCSF *xcsf, LAYER *l)
{
    sam_adapt(xcsf, l->mu, N_MU);
    _Bool mod = false;
    if((l->options & LAYER_EVOLVE_ETA) && neural_layer_mutate_eta(xcsf, l, l->mu[0])) {
        mod = true;
    }
    if(l->options & LAYER_EVOLVE_NEURONS) {
        int n = neural_layer_mutate_neurons(xcsf, l, l->mu[1]);
        if(n != 0) {
            neural_layer_add_neurons(l, n);
            mod = true;
        }
    }
    if((l->options & LAYER_EVOLVE_CONNECT) && neural_layer_mutate_connectivity(l, l->mu[2])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_WEIGHTS) && neural_layer_mutate_weights(l, l->mu[3])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_FUNCTIONS) && neural_layer_mutate_functions(l, l->mu[4])) {
        mod = true;
    }
    return mod;
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
           neural_activation_string(l->function), l->n_inputs, l->n_outputs);
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
    printf("n_active: %d", l->n_active);
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
    s += fwrite(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
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
    if(l->n_inputs < 1 || l->n_outputs < 1 || l->max_outputs < 1 || l->n_weights < 1) {
        printf("neural_layer_connected_load(): read error\n");
        l->n_outputs = 1;
        l->n_weights = 1;
        exit(EXIT_FAILURE);
    }
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->weights = malloc(l->n_weights * sizeof(double));
    l->biases = malloc(l->n_outputs * sizeof(double));
    l->bias_updates = malloc(l->n_outputs * sizeof(double));
    l->weight_updates = malloc(l->n_weights * sizeof(double));
    l->weight_active = malloc(l->n_weights * sizeof(_Bool));
    l->mu = malloc(N_MU * sizeof(double));
    s += fread(l->weights, sizeof(double), l->n_weights, fp);
    s += fread(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fread(l->biases, sizeof(double), l->n_outputs, fp);
    s += fread(l->bias_updates, sizeof(double), l->n_outputs, fp);
    s += fread(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    return s;
}
