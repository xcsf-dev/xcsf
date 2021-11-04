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
#include "neural_layer_upsample.h"
#include "utils.h"

/**
 * @brief Sets a neural network layer's functions to the implementations.
 * @param [in] l The neural network layer to set.
 */
void
layer_set_vptr(struct Layer *l)
{
    switch (l->type) {
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
        case UPSAMPLE:
            l->layer_vptr = &layer_upsample_vtbl;
            break;
        default:
            printf("Error setting layer vptr for type: %d\n", l->type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Mutates the gradient descent rate of a neural layer.
 * @param [in] l The neural network layer to mutate.
 * @param [in] mu The rate of mutation.
 * @return Whether any alterations were made.
 */
bool
layer_mutate_eta(struct Layer *l, const double mu)
{
    const double orig = l->eta;
    l->eta += rand_normal(0, mu);
    l->eta = clamp(l->eta, l->eta_min, l->eta_max);
    if (l->eta != orig) {
        return true;
    }
    return false;
}

/**
 * @brief Returns the number of neurons to add or remove from a layer
 * @param [in] l The neural network layer to mutate.
 * @param [in] mu The rate of mutation.
 * @return The number of neurons to be added or removed.
 */
int
layer_mutate_neurons(const struct Layer *l, const double mu)
{
    int n = 0;
    if (rand_uniform(0, 0.1) < mu) { // 10x higher probability
        while (n == 0) {
            const double m = clamp(rand_normal(0, 0.5), -1, 1);
            n = (int) round(m * l->max_neuron_grow);
        }
        if (l->n_outputs + n < 1) {
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
 * @param [in] l The neural network layer to mutate.
 * @param [in] N The number of neurons to add.
 */
void
layer_add_neurons(struct Layer *l, const int N)
{
    const int old_n_outputs = l->n_outputs;
    const int old_n_weights = l->n_weights;
    l->n_outputs += N;
    l->n_biases = l->n_outputs;
    l->n_weights = l->n_outputs * l->n_inputs;
    layer_guard_outputs(l);
    layer_guard_weights(l);
    l->weights = realloc(l->weights, sizeof(double) * l->n_weights);
    l->weight_active = realloc(l->weight_active, sizeof(bool) * l->n_weights);
    l->weight_updates =
        realloc(l->weight_updates, sizeof(double) * l->n_weights);
    l->state = realloc(l->state, sizeof(double) * l->n_outputs);
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->biases = realloc(l->biases, sizeof(double) * l->n_biases);
    l->bias_updates = realloc(l->bias_updates, sizeof(double) * l->n_biases);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
    for (int i = old_n_weights; i < l->n_weights; ++i) {
        if (l->options & LAYER_EVOLVE_CONNECT && rand_uniform(0, 1) < 0.5) {
            l->weights[i] = 0;
            l->weight_active[i] = false;
        } else {
            l->weights[i] = rand_normal(0, WEIGHT_SD);
            l->weight_active[i] = true;
        }
        l->weight_updates[i] = 0;
    }
    for (int i = old_n_outputs; i < l->n_outputs; ++i) {
        l->biases[i] = 0;
        l->bias_updates[i] = 0;
        l->output[i] = 0;
        l->state[i] = 0;
        l->delta[i] = 0;
    }
    layer_calc_n_active(l);
}

/**
 * @brief Mutates a layer's connectivity by zeroing weights.
 * @param [in] l The neural network layer to mutate.
 * @param [in] mu_enable Probability of enabling a currently disabled weight.
 * @param [in] mu_disable Probability of disabling a currently enabled weight.
 * @return Whether any alterations were made.
 */
bool
layer_mutate_connectivity(struct Layer *l, const double mu_enable,
                          const double mu_disable)
{
    bool mod = false;
    if (l->n_inputs > 1 && l->n_outputs > 1) {
        for (int i = 0; i < l->n_weights; ++i) {
            if (!l->weight_active[i] && rand_uniform(0, 1) < mu_enable) {
                l->weight_active[i] = true;
                l->weights[i] = rand_normal(0, WEIGHT_SD);
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
 * @param [in] l A neural network layer.
 */
void
layer_ensure_input_represention(struct Layer *l)
{
    // each neuron must be connected to at least one input
    for (int i = 0; i < l->n_outputs; ++i) {
        int active = 0;
        const int offset = l->n_inputs * i;
        for (int j = 0; j < l->n_inputs; ++j) {
            if (l->weight_active[offset + j]) {
                ++active;
            }
        }
        if (active < 1) {
            const int r = rand_uniform_int(0, l->n_inputs);
            l->weights[offset + r] = rand_normal(0, WEIGHT_SD);
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
            const int offset = l->n_inputs * rand_uniform_int(0, l->n_outputs);
            if (!l->weight_active[offset + i]) {
                l->weights[offset + i] = rand_normal(0, WEIGHT_SD);
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
 * @param [in] l The neural network layer to mutate.
 * @param [in] mu The rate of mutation.
 * @return Whether any alterations were made.
 */
bool
layer_mutate_weights(struct Layer *l, const double mu)
{
    bool mod = false;
    for (int i = 0; i < l->n_weights; ++i) {
        if (l->weight_active[i]) {
            const double orig = l->weights[i];
            l->weights[i] += rand_normal(0, mu);
            l->weights[i] = clamp(l->weights[i], WEIGHT_MIN, WEIGHT_MAX);
            if (l->weights[i] != orig) {
                mod = true;
            }
        }
    }
    for (int i = 0; i < l->n_biases; ++i) {
        const double orig = l->biases[i];
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
 * @param [in] l The neural network layer to mutate.
 * @param [in] mu The rate of mutation.
 * @return Whether any alterations were made.
 */
bool
layer_mutate_functions(struct Layer *l, const double mu)
{
    bool mod = false;
    if (rand_uniform(0, 1) < mu) {
        const int orig = l->function;
        l->function = rand_uniform_int(0, NUM_ACTIVATIONS);
        if (l->function != orig) {
            mod = true;
        }
    }
    if (l->type == LSTM && rand_uniform(0, 1) < mu) {
        const int orig = l->recurrent_function;
        l->recurrent_function = rand_uniform_int(0, NUM_ACTIVATIONS);
        if (l->recurrent_function != orig) {
            mod = true;
        }
    }
    return mod;
}

/**
 * @brief Prints a layer's weights and biases.
 * @param [in] l The neural network layer to print.
 * @param [in] print_weights Whether to print each individual weight and bias.
 */
void
layer_weight_print(const struct Layer *l, const bool print_weights)
{
    printf("%s\n", layer_weight_json(l, print_weights));
}

/**
 * @brief Returns a json formatted string representation of a layer's weights.
 * @param [in] l The layer to return.
 * @param [in] return_weights Whether to return the values of weights and
 * biases.
 * @return String encoded in json format.
 */
const char *
layer_weight_json(const struct Layer *l, const bool return_weights)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "n_weights", l->n_weights);
    if (return_weights) {
        cJSON *weights = cJSON_CreateDoubleArray(l->weights, l->n_weights);
        cJSON_AddItemToObject(json, "weights", weights);
    }
    cJSON_AddNumberToObject(json, "n_biases", l->n_biases);
    if (return_weights) {
        cJSON *biases = cJSON_CreateDoubleArray(l->biases, l->n_biases);
        cJSON_AddItemToObject(json, "biases", biases);
    }
    cJSON_AddNumberToObject(json, "n_active", l->n_active);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Randomises a layer's weights and biases.
 * @param [in] l The neural network layer to randomise.
 */
void
layer_weight_rand(struct Layer *l)
{
    l->n_active = l->n_weights;
    for (int i = 0; i < l->n_weights; ++i) {
        l->weights[i] = rand_normal(0, WEIGHT_SD_RAND);
        l->weight_active[i] = true;
    }
    for (int i = 0; i < l->n_biases; ++i) {
        l->biases[i] = rand_normal(0, WEIGHT_SD_RAND);
    }
}

/**
 * @brief Clamps a layer's weights and biases in range [WEIGHT_MIN, WEIGHT_MAX].
 * @param [in] l The neural network layer to clamp.
 */
void
layer_weight_clamp(const struct Layer *l)
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
 * @brief Recalculates the number of active connections within a layer.
 * @param [in] l The layer to recalculate the number of active connections.
 */
void
layer_calc_n_active(struct Layer *l)
{
    l->n_active = 0;
    for (int i = 0; i < l->n_weights; ++i) {
        if (l->weight_active[i]) {
            ++(l->n_active);
        }
    }
}

/**
 * @brief Initialises a layer's gradient descent rate.
 * @param [in] l The layer to initialise.
 */
void
layer_init_eta(struct Layer *l)
{
    if (l->options & LAYER_EVOLVE_ETA) {
        l->eta = rand_uniform(l->eta_min, l->eta_max);
    } else {
        l->eta = l->eta_max;
    }
}

/**
 * @brief Initialises a layer to default values.
 * @param [in] l The layer to initialise.
 */
void
layer_defaults(struct Layer *l)
{
    l->type = 0;
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
    l->eta_max = 0;
    l->eta_min = 0;
    l->momentum = 0;
    l->decay = 0;
    l->n_inputs = 0;
    l->n_outputs = 0;
    l->max_outputs = 0;
    l->max_neuron_grow = 0;
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
}

/**
 * @brief Returns a string representation of a layer type from an integer.
 * @param [in] type Integer representation of a layer type.
 * @return String representing the name of the layer type.
 */
const char *
layer_type_as_string(const int type)
{
    switch (type) {
        case CONNECTED:
            return STRING_CONNECTED;
        case DROPOUT:
            return STRING_DROPOUT;
        case NOISE:
            return STRING_NOISE;
        case SOFTMAX:
            return STRING_SOFTMAX;
        case RECURRENT:
            return STRING_RECURRENT;
        case LSTM:
            return STRING_LSTM;
        case MAXPOOL:
            return STRING_MAXPOOL;
        case CONVOLUTIONAL:
            return STRING_CONVOLUTIONAL;
        case AVGPOOL:
            return STRING_AVGPOOL;
        case UPSAMPLE:
            return STRING_UPSAMPLE;
        default:
            printf("layer_type_as_string(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns a whether a layer type expects images as input.
 * @param [in] type Integer representation of a layer type.
 * @return Whether images (height × width × channels) are expected as inputs.
 */
bool
layer_receives_images(const int type)
{
    switch (type) {
        case AVGPOOL:
        case MAXPOOL:
        case UPSAMPLE:
        case CONVOLUTIONAL:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Returns the integer representation of a layer type given a name.
 * @param [in] type String representation of a layer type.
 * @return Integer representing the layer type.
 */
int
layer_type_as_int(const char *type)
{
    if (strncmp(type, STRING_CONNECTED, 10) == 0) {
        return CONNECTED;
    }
    if (strncmp(type, STRING_DROPOUT, 8) == 0) {
        return DROPOUT;
    }
    if (strncmp(type, STRING_SOFTMAX, 8) == 0) {
        return SOFTMAX;
    }
    if (strncmp(type, STRING_NOISE, 6) == 0) {
        return NOISE;
    }
    if (strncmp(type, STRING_RECURRENT, 9) == 0) {
        return RECURRENT;
    }
    if (strncmp(type, STRING_LSTM, 5) == 0) {
        return LSTM;
    }
    if (strncmp(type, STRING_MAXPOOL, 8) == 0) {
        return MAXPOOL;
    }
    if (strncmp(type, STRING_CONVOLUTIONAL, 14) == 0) {
        return CONVOLUTIONAL;
    }
    if (strncmp(type, STRING_AVGPOOL, 8) == 0) {
        return AVGPOOL;
    }
    if (strncmp(type, STRING_UPSAMPLE, 9) == 0) {
        return UPSAMPLE;
    }
    printf("layer_type_as_int(): invalid type: %s\n", type);
    exit(EXIT_FAILURE);
}

/**
 * @brief Check number of biases is within bounds.
 * @param [in] l Layer to check.
 */
void
layer_guard_biases(const struct Layer *l)
{
    if (l->n_biases < 1 || l->n_biases > N_OUTPUTS_MAX) {
        printf("layer_guard_biases() invalid size\n");
        layer_print(l, false);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Check number of outputs is within bounds.
 * @param [in] l Layer to check.
 */
void
layer_guard_outputs(const struct Layer *l)
{
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX) {
        printf("layer_guard_outputs() invalid size\n");
        layer_print(l, false);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Check number of weights is within bounds.
 * @param [in] l Layer to check.
 */
void
layer_guard_weights(const struct Layer *l)
{
    if (l->n_weights < 1 || l->n_weights > N_WEIGHTS_MAX) {
        printf("layer_guard_weights() invalid size\n");
        layer_print(l, false);
        exit(EXIT_FAILURE);
    }
}
