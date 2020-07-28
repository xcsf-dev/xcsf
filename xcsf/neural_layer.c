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
 * @file neural_layer.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Interface for neural network layers.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_noise.h"
#include "neural_layer_recurrent.h"
#include "neural_layer_lstm.h"
#include "neural_layer_softmax.h"
#include "neural_layer_maxpool.h"

#define ETA_MIN 0.000001 //!< Minimum gradient descent rate
#define WEIGHT_MIN -10 //!< Minimum value of a weight or bias
#define WEIGHT_MAX 10 //!< Maximum value of a weight or bias

/**
 * @brief Sets a neural network layer's functions to the implementations.
 * @param l The neural network layer to set.
 */
void layer_set_vptr(LAYER *l)
{
    switch(l->layer_type) {
        case CONNECTED:
            l->layer_vptr = &layer_connected_vtbl;
            break;
        case DROPOUT:
            l->layer_vptr = &layer_dropout_vtbl;
            break;
        case NOISE:
            l->layer_vptr = &layer_noise_vtbl;
            break;
        case RECURRENT:
            l->layer_vptr = &layer_recurrent_vtbl;
            break;
        case LSTM:
            l->layer_vptr = &layer_lstm_vtbl;
            break;
        case SOFTMAX:
            l->layer_vptr = &layer_softmax_vtbl;
            break;
        case MAXPOOL:
            l->layer_vptr = &layer_maxpool_vtbl;
            break;
        default:
            printf("Error setting layer vptr for type: %d\n", l->layer_type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Mutates the gradient descent rate of a neural layer.
 * @param xcsf The XCSF data structure.
 * @param l The neural network layer to mutate.
 * @param mu The rate of mutation.
 * @return Whether any alterations were made.
 */
_Bool layer_mutate_eta(const XCSF *xcsf, LAYER *l, double mu)
{
    double orig = l->eta;
    l->eta += rand_normal(0, mu);
    l->eta = clamp(ETA_MIN, xcsf->PRED_ETA, l->eta);
    if(l->eta != orig) {
        return true;
    }
    return false;
}

int layer_mutate_neurons(const XCSF *xcsf, const LAYER *l, double mu)
{
    int n = (int) round(((2 * mu) - 1) * xcsf->MAX_NEURON_GROW);
    if(n < 0 && l->n_outputs + n < 1) {
        n = -(l->n_outputs - 1);
    } else if(l->n_outputs + n > l->max_outputs) {
        n = l->max_outputs - l->n_outputs;
    }
    return n;
}

void layer_add_neurons(LAYER *l, int n)
{
    // assumes n is appropriately bounds checked
    // negative n will remove neurons
    int n_outputs = l->n_outputs + n;
    int n_weights = n_outputs * l->n_inputs;
    size_t w_size_t = n_weights * sizeof(double);
    size_t o_size_t = n_outputs * sizeof(double);
    l->weights = (double*) realloc(l->weights, w_size_t);
    l->weight_active = (_Bool*) realloc(l->weight_active, n_weights * sizeof(_Bool));
    l->weight_updates = (double*) realloc(l->weight_updates, w_size_t);
    l->state = (double*) realloc(l->state, o_size_t);
    l->output = (double*) realloc(l->output, o_size_t);
    l->biases = (double*) realloc(l->biases, o_size_t);
    l->bias_updates = (double*) realloc(l->bias_updates, o_size_t);
    l->delta = (double*) realloc(l->delta, o_size_t);
    if(n > 0) {
        for(int i = l->n_weights; i < n_weights; i++) {
            if(l->options & LAYER_EVOLVE_CONNECT && rand_uniform(0, 1) < 0.5) {
                l->weights[i] = 0;
                l->weight_active[i] = false;
            } else {
                l->weights[i] = rand_normal(0, 0.1);
                l->weight_active[i] = true;
                l->n_active += 1;
            }
            l->weight_updates[i] = 0;
        }
        for(int i = l->n_outputs; i < n_outputs; i++) {
            l->biases[i] = 0;
            l->bias_updates[i] = 0;
            l->output[i] = 0;
            l->state[i] = 0;
            l->delta[i] = 0;
        }
    }
    l->n_weights = n_weights;
    l->n_outputs = n_outputs;
}

_Bool layer_mutate_connectivity(LAYER *l, double mu)
{
    if(l->n_inputs < 2) {
        return false;
    }
    _Bool mod = false;
    for(int i = 0; i < l->n_weights; i++) {
        if(rand_uniform(0, 1) < mu) {
            l->weight_active[i] = ! l->weight_active[i];
            if(l->weight_active[i]) {
                l->weights[i] = rand_normal(0, 0.1);
                l->n_active += 1;
            } else {
                l->weights[i] = 0;
                l->n_active -= 1;
            }
            mod = true;
        }
    }
    // at least one connection must be active
    if(l->n_active < 1) {
        int r = irand_uniform(0, l->n_weights);
        l->weights[r] = rand_normal(0, 0.1);
        l->weight_active[r] = true;
        l->n_active += 1;
    }
    return mod;
}

_Bool layer_mutate_weights(LAYER *l, double mu)
{
    _Bool mod = false;
    for(int i = 0; i < l->n_weights; i++) {
        if(l->weight_active[i]) {
            double orig = l->weights[i];
            l->weights[i] += rand_normal(0, mu);
            l->weights[i] = clamp(WEIGHT_MIN, WEIGHT_MAX, l->weights[i]);
            if(l->weights[i] != orig) {
                mod = true;
            }
        }
    }
    for(int i = 0; i < l->n_outputs; i++) {
        double orig = l->biases[i];
        l->biases[i] += rand_normal(0, mu);
        l->biases[i] = clamp(WEIGHT_MIN, WEIGHT_MAX, l->biases[i]);
        if(l->biases[i] != orig) {
            mod = true;
        }
    }
    return mod;
}

_Bool layer_mutate_functions(LAYER *l, double mu)
{
    _Bool mod = false;
    if(rand_uniform(0, 1) < mu) {
        int orig = l->function;
        l->function = irand_uniform(0, NUM_ACTIVATIONS);
        if(l->function != orig) {
            mod = true;
        }
    }
    if(l->layer_type == LSTM && rand_uniform(0, 1) < mu) {
        int orig = l->recurrent_function;
        l->recurrent_function = irand_uniform(0, NUM_ACTIVATIONS);
        if(l->recurrent_function != orig) {
            mod = true;
        }
    }
    return mod;
}

void layer_weight_clamp(const LAYER *l)
{
    for(int i = 0; i < l->n_weights; i++) {
        if(l->weight_active[i]) {
            l->weights[i] = clamp(WEIGHT_MIN, WEIGHT_MAX, l->weights[i]);
        } else {
            l->weights[i] = 0;
        }
    }
    for(int i = 0; i < l->n_outputs; i++) {
        l->biases[i] = clamp(WEIGHT_MIN, WEIGHT_MAX, l->biases[i]);
    }
}

void layer_calc_n_active(LAYER *l)
{
    l->n_active = l->n_weights;
    for(int i = 0; i < l->n_weights; i++) {
        if(l->weights[i] == 0) {
            l->n_active -= 1;
        }
    }
}

void layer_init_eta(const XCSF *xcsf, LAYER *l)
{
    l->eta = rand_uniform(ETA_MIN, xcsf->PRED_ETA);
}
