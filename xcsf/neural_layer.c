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

#include "neural_activations.h"
#include "neural_layer_avgpool.h"
#include "neural_layer_connected.h"
#include "neural_layer_convolutional.h"
#include "neural_layer_dropout.h"
#include "neural_layer_lstm.h"
#include "neural_layer_maxpool.h"
#include "neural_layer_noise.h"
#include "neural_layer_recurrent.h"
#include "neural_layer_softmax.h"
#include "utils.h"

/**
 * @brief Sets a neural network layer's functions to the implementations.
 * @param l The neural network layer to set.
 */
void
layer_set_vptr(struct LAYER *l)
{
    switch (l->layer_type) {
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
        case CONVOLUTIONAL:
            l->layer_vptr = &layer_convolutional_vtbl;
            break;
        case AVGPOOL:
            l->layer_vptr = &layer_avgpool_vtbl;
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
_Bool
layer_mutate_eta(const struct XCSF *xcsf, struct LAYER *l, double mu)
{
    double orig = l->eta;
    l->eta += rand_normal(0, mu);
    l->eta = clamp(l->eta, ETA_MIN, xcsf->PRED_ETA);
    if (l->eta != orig) {
        return true;
    }
    return false;
}

/**
 * @brief Returns the number of neurons to add or remove from a layer
 * @param xcsf The XCSF data structure.
 * @param l The neural network layer to mutate.
 * @return The number of neurons to be added or removed.
 */
int
layer_mutate_neurons(const struct XCSF *xcsf, const struct LAYER *l)
{
    int n = 0;
    if (rand_uniform(0, 1) < 0.3333) {
        double m = clamp(rand_normal(0, 0.5), -1, 1);
        n = (int) round(m * xcsf->MAX_NEURON_GROW);
        if (n < 0 && l->n_outputs + n < 1) {
            n = -(l->n_outputs - 1);
        } else if (l->n_outputs + n > l->max_outputs) {
            n = l->max_outputs - l->n_outputs;
        }
    }
    return n;
}

/**
 * @brief Adds N neurons to a layer. Negative N removes neurons.
 * @pre N must be appropriately bounds checked for the layer.
 * @param l The neural network layer to mutate.
 * @param N The number of neurons to add.
 */
void
layer_add_neurons(struct LAYER *l, int N)
{
    int n_outputs = l->n_outputs + N;
    int n_weights = n_outputs * l->n_inputs;
    l->weights = realloc(l->weights, sizeof(double) * n_weights);
    l->weight_active = realloc(l->weight_active, sizeof(_Bool) * n_weights);
    l->weight_updates = realloc(l->weight_updates, sizeof(double) * n_weights);
    l->state = realloc(l->state, sizeof(double) * n_outputs);
    l->output = realloc(l->output, sizeof(double) * n_outputs);
    l->biases = realloc(l->biases, sizeof(double) * n_outputs);
    l->bias_updates = realloc(l->bias_updates, sizeof(double) * n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * n_outputs);
    if (N > 0) {
        for (int i = l->n_weights; i < n_weights; ++i) {
            if (l->options & LAYER_EVOLVE_CONNECT && rand_uniform(0, 1) < 0.5) {
                l->weights[i] = 0;
                l->weight_active[i] = false;
            } else {
                l->weights[i] = rand_normal(0, 0.1);
                l->weight_active[i] = true;
            }
            l->weight_updates[i] = 0;
        }
        for (int i = l->n_outputs; i < n_outputs; ++i) {
            l->biases[i] = 0;
            l->bias_updates[i] = 0;
            l->output[i] = 0;
            l->state[i] = 0;
            l->delta[i] = 0;
        }
    }
    l->n_weights = n_weights;
    l->n_outputs = n_outputs;
    l->n_biases = n_outputs;
    layer_calc_n_active(l);
}

/**
 * @brief Mutates a layer's connectivity by zeroing weights.
 * @param l The neural network layer to mutate.
 * @param mu_enable The probability of enabling a currently disabled weight.
 * @param mu_disable The probability of disabling a currently enabled weight.
 * @return Whether any alterations were made.
 */
_Bool
layer_mutate_connectivity(struct LAYER *l, double mu_enable, double mu_disable)
{
    _Bool mod = false;
    if (l->n_inputs > 1 && l->n_outputs > 1) {
        for (int i = 0; i < l->n_weights; ++i) {
            if (!l->weight_active[i] && rand_uniform(0, 1) < mu_enable) {
                l->weight_active[i] = true;
                l->weights[i] = rand_normal(0, 0.1);
                ++(l->n_active);
                mod = true;
            } else if (l->weight_active[i] && rand_uniform(0, 1) < mu_disable) {
                l->weight_active[i] = false;
                l->weights[i] = 0;
                --(l->n_active);
                mod = true;
            }
        }
    }
    return mod;
}

/**
 * @brief Ensures that each neuron is connected to at least one input and each
 * input is connected to at least one neuron.
 * @param l A neural network layer.
 */
void
layer_ensure_input_represention(struct LAYER *l)
{
    // each neuron must be connected to at least one input
    for (int i = 0; i < l->n_outputs; ++i) {
        int active = 0;
        int offset = l->n_inputs * i;
        for (int j = 0; j < l->n_inputs; ++j) {
            if (l->weight_active[offset + j]) {
                ++active;
            }
        }
        if (active < 1) {
            int r = irand_uniform(0, l->n_inputs);
            l->weights[offset + r] = rand_normal(0, 0.1);
            l->weight_active[offset + r] = true;
            ++(l->n_active);
            ++active;
        }
    }
    // each input must be represented at least once
    for (int i = 0; i < l->n_inputs; ++i) {
        int active = 0;
        for (int j = 0; j < l->n_outputs; ++j) {
            if (l->weight_active[l->n_inputs * j + i]) {
                ++active;
            }
        }
        while (active < 1) {
            int offset = l->n_inputs * irand_uniform(0, l->n_outputs);
            if (!l->weight_active[offset + i]) {
                l->weights[offset + i] = rand_normal(0, 0.1);
                l->weight_active[offset + i] = true;
                ++(l->n_active);
                ++active;
            }
        }
    }
}

/**
 * @brief Mutates a layer's weights and biases by adding random numbers from a
 * Gaussian normal distribution with zero mean and standard deviation equal to
 * the mutation rate.
 * @param l The neural network layer to mutate.
 * @param mu The rate of mutation.
 * @return Whether any alterations were made.
 */
_Bool
layer_mutate_weights(struct LAYER *l, double mu)
{
    _Bool mod = false;
    for (int i = 0; i < l->n_weights; ++i) {
        if (l->weight_active[i]) {
            double orig = l->weights[i];
            l->weights[i] += rand_normal(0, mu);
            l->weights[i] = clamp(l->weights[i], WEIGHT_MIN, WEIGHT_MAX);
            if (l->weights[i] != orig) {
                mod = true;
            }
        }
    }
    for (int i = 0; i < l->n_biases; ++i) {
        double orig = l->biases[i];
        l->biases[i] += rand_normal(0, mu);
        l->biases[i] = clamp(l->biases[i], WEIGHT_MIN, WEIGHT_MAX);
        if (l->biases[i] != orig) {
            mod = true;
        }
    }
    return mod;
}

/**
 * @brief Mutates a layer's activation function by random selection.
 * @param l The neural network layer to mutate.
 * @param mu The rate of mutation.
 * @return Whether any alterations were made.
 */
_Bool
layer_mutate_functions(struct LAYER *l, double mu)
{
    _Bool mod = false;
    if (rand_uniform(0, 1) < mu) {
        int orig = l->function;
        l->function = irand_uniform(0, NUM_ACTIVATIONS);
        if (l->function != orig) {
            mod = true;
        }
    }
    if (l->layer_type == LSTM && rand_uniform(0, 1) < mu) {
        int orig = l->recurrent_function;
        l->recurrent_function = irand_uniform(0, NUM_ACTIVATIONS);
        if (l->recurrent_function != orig) {
            mod = true;
        }
    }
    return mod;
}

/**
 * @brief Prints a layer's weights and biases.
 * @param l The neural network layer to print.
 * @param print_weights Whether to print each individual weight and bias.
 */
void
layer_weight_print(const struct LAYER *l, _Bool print_weights)
{
    printf("weights (%d): ", l->n_weights);
    if (print_weights) {
        for (int i = 0; i < l->n_weights; ++i) {
            printf("%.4f, ", l->weights[i]);
        }
    }
    printf("biases (%d): ", l->n_biases);
    if (print_weights) {
        for (int i = 0; i < l->n_biases; ++i) {
            printf("%.4f, ", l->biases[i]);
        }
    }
    printf("n_active: %d", l->n_active);
}

/**
 * @brief Randomises a layer's weights and biases.
 * @param xcsf The XCSF data structure.
 * @param l The neural network layer to randomise.
 */
void
layer_weight_rand(const struct XCSF *xcsf, struct LAYER *l)
{
    (void) xcsf;
    l->n_active = l->n_weights;
    for (int i = 0; i < l->n_weights; ++i) {
        l->weights[i] = rand_normal(0, 1);
        l->weight_active[i] = true;
    }
    for (int i = 0; i < l->n_biases; ++i) {
        l->biases[i] = rand_normal(0, 1);
    }
}

/**
 * @brief Clamps a layer's weights and biases in range [WEIGHT_MIN, WEIGHT_MAX].
 * @param l The neural network layer to clamp.
 */
void
layer_weight_clamp(const struct LAYER *l)
{
    for (int i = 0; i < l->n_weights; ++i) {
        if (l->weight_active[i]) {
            l->weights[i] = clamp(l->weights[i], WEIGHT_MIN, WEIGHT_MAX);
        } else {
            l->weights[i] = 0;
        }
    }
    for (int i = 0; i < l->n_biases; ++i) {
        l->biases[i] = clamp(l->biases[i], WEIGHT_MIN, WEIGHT_MAX);
    }
}

/**
 * @brief Sets n_active to the number of non-zero weights within a layer.
 * @param l The layer to calculate the number of non-zero weights.
 */
void
layer_calc_n_active(struct LAYER *l)
{
    l->n_active = l->n_weights;
    for (int i = 0; i < l->n_weights; ++i) {
        if (l->weights[i] == 0) {
            --(l->n_active);
        }
    }
}

/**
 * @brief Initialises a layer's gradient descent rate.
 * @param xcsf The XCSF data structure.
 * @param l The layer to initialise.
 */
void
layer_init_eta(const struct XCSF *xcsf, struct LAYER *l)
{
    if (l->options & LAYER_EVOLVE_ETA) {
        l->eta = rand_uniform(ETA_MIN, xcsf->PRED_ETA);
    } else {
        l->eta = xcsf->PRED_ETA;
    }
}

/**
 * @brief Initialises a layer to default values.
 * @param l The layer to initialise.
 */
void
layer_init(struct LAYER *l)
{
    l->layer_type = 0;
    l->state = NULL;
    l->output = NULL;
    l->options = 0;
    l->weights = NULL;
    l->weight_active = NULL;
    l->biases = NULL;
    l->bias_updates = NULL;
    l->weight_updates = NULL;
    l->delta = NULL;
    l->mu = NULL;
    l->eta = 0;
    l->n_inputs = 0;
    l->n_outputs = 0;
    l->max_outputs = 0;
    l->n_weights = 0;
    l->n_biases = 0;
    l->n_active = 0;
    l->function = 0;
    l->scale = 0;
    l->probability = 0;
    l->layer_vptr = NULL;
    l->prev_state = NULL;
    l->input_layer = NULL;
    l->self_layer = NULL;
    l->output_layer = NULL;
    l->recurrent_function = 0;
    l->uf = NULL;
    l->ui = NULL;
    l->ug = NULL;
    l->uo = NULL;
    l->wf = NULL;
    l->wi = NULL;
    l->wg = NULL;
    l->wo = NULL;
    l->cell = NULL;
    l->prev_cell = NULL;
    l->f = NULL;
    l->i = NULL;
    l->g = NULL;
    l->o = NULL;
    l->c = NULL;
    l->h = NULL;
    l->temp = NULL;
    l->temp2 = NULL;
    l->temp3 = NULL;
    l->dc = NULL;
    l->height = 0;
    l->width = 0;
    l->channels = 0;
    l->pad = 0;
    l->out_w = 0;
    l->out_h = 0;
    l->out_c = 0;
    l->size = 0;
    l->stride = 0;
    l->indexes = NULL;
    l->n_filters = 0;
    l->workspace_size = 0;
}
