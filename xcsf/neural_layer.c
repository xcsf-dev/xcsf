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
#include "neural_layer_softmax.h"

#define ETA_MIN 0.000001 //!< Minimum gradient descent rate
#define WEIGHT_MIN -10 //!< Minimum value of a weight or bias
#define WEIGHT_MAX 10 //!< Maximum value of a weight or bias

/**
 * @brief Sets a neural network layer's functions to the implementations.
 * @param l The neural network layer to set.
 */
void neural_layer_set_vptr(LAYER *l)
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
        case SOFTMAX:
            l->layer_vptr = &layer_softmax_vtbl;
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
_Bool neural_layer_mutate_eta(const XCSF *xcsf, LAYER *l, double mu)
{
    double orig = l->eta;
    l->eta += rand_normal(0, mu);
    l->eta = clamp(ETA_MIN, xcsf->PRED_ETA, l->eta);
    if(l->eta != orig) {
        return true;
    }
    return false;
}

int neural_layer_mutate_neurons(const XCSF *xcsf, LAYER *l, double mu)
{
    int n = (int) round(((2 * mu) - 1) * xcsf->MAX_NEURON_MOD);
    if(n < 0 && l->n_outputs + n < 1) {
        n = -(l->n_outputs - 1);
    } else if(l->n_outputs + n > l->max_outputs) {
        n = l->max_outputs - l->n_outputs;
    }
    return n;
}

void neural_layer_add_neurons(LAYER *l, int n)
{
    // assumes n is appropriately bounds checked
    // negative n will remove neurons
    int n_outputs = l->n_outputs + n;
    int n_weights = n_outputs * l->n_inputs;
    double *weights = malloc(n_weights * sizeof(double));
    _Bool *weight_active = malloc(n_weights * sizeof(_Bool));
    double *weight_updates = malloc(n_weights * sizeof(double));
    double *state = calloc(n_outputs, sizeof(double));
    double *output = calloc(n_outputs, sizeof(double));
    double *biases = malloc(n_outputs * sizeof(double));
    double *bias_updates = malloc(n_outputs * sizeof(double));
    double *delta = calloc(n_outputs, sizeof(double));
    int w_len = n_weights;
    int o_len = n_outputs;
    if(n > 0) {
        w_len = l->n_weights;
        o_len = l->n_outputs;
    }
    memcpy(weights, l->weights, w_len * sizeof(double));
    memcpy(weight_active, l->weight_active, w_len * sizeof(_Bool));
    memcpy(weight_updates, l->weight_updates, w_len * sizeof(double));
    memcpy(biases, l->biases, o_len * sizeof(double));
    memcpy(bias_updates, l->bias_updates, o_len * sizeof(double));
    if(n > 0) {
        for(int i = l->n_weights; i < n_weights; i++) {
            if(l->options & LAYER_EVOLVE_CONNECT && rand_uniform(0, 1) < 0.5) {
                weights[i] = 0;
                weight_active[i] = false;
            } else {
                weights[i] = rand_normal(0, 0.1);
                weight_active[i] = true;
            }
            weight_updates[i] = 0;
        }
        for(int i = l->n_outputs; i < n_outputs; i++) {
            biases[i] = 0;
            bias_updates[i] = 0;
        }
    }
    free(l->weights);
    free(l->weight_active);
    free(l->weight_updates);
    free(l->state);
    free(l->output);
    free(l->biases);
    free(l->bias_updates);
    free(l->delta);
    l->weights = weights;
    l->weight_active = weight_active;
    l->weight_updates = weight_updates;
    l->state = state;
    l->output = output;
    l->biases = biases;
    l->bias_updates = bias_updates;
    l->delta = delta;
    l->n_weights = n_weights;
    l->n_outputs = n_outputs;
    neural_layer_calc_n_active(l);
    // at least one connection must be active
    if(l->n_active == 0) {
        int r = irand_uniform(0, l->n_weights);
        weights[r] = rand_normal(0, 0.1);
        l->weight_active[r] = true;
        l->n_active += 1;
    }
}

_Bool neural_layer_mutate_connectivity(LAYER *l, double mu)
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
    return mod;
}

_Bool neural_layer_mutate_weights(LAYER *l, double mu)
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

_Bool neural_layer_mutate_functions(LAYER *l, double mu)
{
    if(rand_uniform(0, 1) < mu) {
        int orig = l->function;
        l->function = irand_uniform(0, NUM_ACTIVATIONS);
        if(l->function != orig) {
            return true;
        }
    }
    return false;
}

void neural_layer_weight_clamp(const LAYER *l)
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

void neural_layer_calc_n_active(LAYER *l)
{
    l->n_active = l->n_weights;
    for(int i = 0; i < l->n_weights; i++) {
        if(l->weights[i] == 0) {
            l->n_active -= 1;
        }
    }
}

void neural_layer_init_eta(const XCSF *xcsf, LAYER *l)
{
    l->eta = rand_uniform(ETA_MIN, xcsf->PRED_ETA);
}
