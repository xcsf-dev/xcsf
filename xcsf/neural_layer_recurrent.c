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
 * @file neural_layer_recurrent.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a recurrent layer of perceptrons.
 * @details Fully-connected with a step of 1.
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
#include "neural_layer_recurrent.h"

#define N_MU 5 //!< Number of mutation rates applied to a recurrent layer

LAYER *neural_layer_recurrent_init(const XCSF *xcsf, int in, int n_init, int n_max, int f,
                                   uint32_t o)
{
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = RECURRENT;
    l->layer_vptr = &layer_recurrent_vtbl;
    l->options = o;
    l->function = f;
    l->n_inputs = in;
    l->n_outputs = n_init;
    l->max_outputs = n_max;
    l->state = calloc(l->n_outputs, sizeof(double));
    l->input_layer = neural_layer_connected_init(xcsf, in, n_init, n_max, LINEAR, o);
    l->self_layer = neural_layer_connected_init(xcsf, n_init, n_init, n_max, LINEAR, o);
    l->output_layer = neural_layer_connected_init(xcsf, n_init, n_init, n_max, f, o);
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;
    l->prev_state = calloc(l->n_outputs, sizeof(double));
    l->n_active = l->input_layer->n_active + l->self_layer->n_active
                  + l->output_layer->n_active;
    // one set of mutation rates for the whole layer
    l->mu = malloc(N_MU * sizeof(double));
    sam_init(xcsf, l->mu, N_MU);
    // one gradient descent rate for the whole layer
    l->eta = l->input_layer->eta;
    l->self_layer->eta = l->eta;
    l->output_layer->eta = l->eta;
    return l;
}

LAYER *neural_layer_recurrent_copy(const XCSF *xcsf, const LAYER *src)
{
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->options = src->options;
    l->function = src->function;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->n_active = src->n_active;
    l->eta = src->eta;
    l->max_outputs = src->max_outputs;
    l->state = calloc(src->n_outputs, sizeof(double));
    l->input_layer = neural_layer_connected_copy(xcsf, src->input_layer);
    l->self_layer = neural_layer_connected_copy(xcsf, src->self_layer);
    l->output_layer = neural_layer_connected_copy(xcsf, src->output_layer);
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;
    l->mu = malloc(N_MU * sizeof(double));
    memcpy(l->mu, src->mu, N_MU * sizeof(double));
    l->prev_state = malloc(src->n_outputs * sizeof(double));
    memcpy(l->prev_state, src->prev_state, src->n_outputs * sizeof(double));
    return l;
}

void neural_layer_recurrent_free(const XCSF *xcsf, const LAYER *l)
{
    neural_layer_connected_free(xcsf, l->input_layer);
    neural_layer_connected_free(xcsf, l->self_layer);
    neural_layer_connected_free(xcsf, l->output_layer);
    free(l->state);
    free(l->prev_state);
    free(l->mu);
}

void neural_layer_recurrent_rand(const XCSF *xcsf, LAYER *l)
{
    neural_layer_connected_rand(xcsf, l->input_layer);
    neural_layer_connected_rand(xcsf, l->self_layer);
    neural_layer_connected_rand(xcsf, l->output_layer);
}

void neural_layer_recurrent_forward(const XCSF *xcsf, const LAYER *l, NET *net)
{
    memcpy(l->prev_state, l->state, l->n_outputs * sizeof(double));
    neural_layer_connected_forward(xcsf, l->input_layer, net);
    net->input = l->output_layer->output;
    neural_layer_connected_forward(xcsf, l->self_layer, net);
    memcpy(l->state, l->input_layer->output, l->n_outputs * sizeof(double));
    blas_axpy(l->n_outputs, 1, l->self_layer->output, 1, l->state, 1);
    net->input = l->state;
    neural_layer_connected_forward(xcsf, l->output_layer, net);
}

void neural_layer_recurrent_backward(const XCSF *xcsf, const LAYER *l, NET *net)
{
    memset(l->input_layer->delta, 0, l->n_outputs * sizeof(double));
    memset(l->self_layer->delta, 0,  l->n_outputs * sizeof(double));
    const double *input = net->input;
    double *delta = net->delta;
    net->input = l->state;
    net->delta = l->self_layer->delta;
    neural_layer_connected_backward(xcsf, l->output_layer, net);
    memcpy(l->input_layer->delta, l->self_layer->delta, l->n_outputs * sizeof(double));
    net->input = l->prev_state;
    net->delta = 0;
    neural_layer_connected_backward(xcsf, l->self_layer, net);
    net->input = input;
    net->delta = delta;
    neural_layer_connected_backward(xcsf, l->input_layer, net);
}

void neural_layer_recurrent_update(const XCSF *xcsf, const LAYER *l)
{
    if(l->options & LAYER_SGD_WEIGHTS) {
        neural_layer_connected_update(xcsf, l->input_layer);
        neural_layer_connected_update(xcsf, l->self_layer);
        neural_layer_connected_update(xcsf, l->output_layer);
    }
}

void neural_layer_recurrent_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    neural_layer_connected_resize(xcsf, l->input_layer, prev);
    l->n_inputs = l->input_layer->n_inputs;
    l->n_active = l->input_layer->n_active + l->self_layer->n_active
                  + l->output_layer->n_active;
}

double *neural_layer_recurrent_output(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    return l->output;
}

_Bool neural_layer_recurrent_mutate(const XCSF *xcsf, LAYER *l)
{
    sam_adapt(xcsf, l->mu, N_MU);
    _Bool mod = false;
    if((l->options & LAYER_EVOLVE_ETA) && neural_layer_mutate_eta(xcsf, l, l->mu[0])) {
        l->input_layer->eta = l->eta;
        l->self_layer->eta = l->eta;
        l->output_layer->eta = l->eta;
        mod = true;
    }
    if(l->options & LAYER_EVOLVE_NEURONS) {
        int n = neural_layer_mutate_neurons(xcsf, l->self_layer, l->mu[1]);
        if(n != 0) {
            neural_layer_add_neurons(l->input_layer, n);
            neural_layer_add_neurons(l->self_layer, n);
            neural_layer_add_neurons(l->output_layer, n);
            mod = true;
        }
    }
    if(l->options & LAYER_EVOLVE_CONNECT) {
        mod = neural_layer_mutate_connectivity(l->input_layer, l->mu[2]) ? true : mod;
        mod = neural_layer_mutate_connectivity(l->self_layer, l->mu[2]) ? true : mod;
        mod = neural_layer_mutate_connectivity(l->output_layer, l->mu[2]) ? true : mod;
        l->n_active = l->input_layer->n_active + l->self_layer->n_active
                      + l->output_layer->n_active;
    }
    if(l->options & LAYER_EVOLVE_WEIGHTS) {
        mod = neural_layer_mutate_weights(l->input_layer, l->mu[3]) ? true : mod;
        mod = neural_layer_mutate_weights(l->self_layer, l->mu[3]) ? true : mod;
        mod = neural_layer_mutate_weights(l->output_layer, l->mu[3]) ? true : mod;
    }
    if(l->options & LAYER_EVOLVE_FUNCTIONS && neural_layer_mutate_functions(l, l->mu[4])) {
        l->input_layer->function = l->function;
        l->self_layer->function = l->function;
        l->output_layer->function = l->function;
        mod = true;
    }
    return mod;
}

void neural_layer_recurrent_print(const XCSF *xcsf, const LAYER *l, _Bool print_weights)
{
    printf("recurrent %s, in = %d, out = %d\n",
           neural_activation_string(l->function), l->n_inputs, l->n_outputs);
    if(print_weights) {
        printf("recurrent input layer:\n");
        neural_layer_connected_print(xcsf, l->input_layer, print_weights);
        printf("recurrent self layer:\n");
        neural_layer_connected_print(xcsf, l->self_layer, print_weights);
        printf("recurrent output layer:\n");
        neural_layer_connected_print(xcsf, l->output_layer, print_weights);
    }
}

size_t neural_layer_recurrent_save(const XCSF *xcsf, const LAYER *l, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    s += fwrite(l->state, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->prev_state, sizeof(double), l->n_outputs, fp);
    s += neural_layer_connected_save(xcsf, l->input_layer, fp);
    s += neural_layer_connected_save(xcsf, l->self_layer, fp);
    s += neural_layer_connected_save(xcsf, l->output_layer, fp);
    return s;
}

size_t neural_layer_recurrent_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    if(l->n_inputs < 1 || l->n_outputs < 1 || l->max_outputs < 1) {
        printf("neural_layer_recurrent_load(): read error\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->mu = malloc(N_MU * sizeof(double));
    l->state = malloc(l->n_outputs * sizeof(double));
    l->prev_state = malloc(l->n_outputs * sizeof(double));
    s += fread(l->mu, sizeof(double), N_MU, fp);
    s += fread(l->state, sizeof(double), l->n_outputs, fp);
    s += fread(l->prev_state, sizeof(double), l->n_outputs, fp);
    s += neural_layer_connected_load(xcsf, l->input_layer, fp);
    s += neural_layer_connected_load(xcsf, l->self_layer, fp);
    s += neural_layer_connected_load(xcsf, l->output_layer, fp);
    return s;
}
