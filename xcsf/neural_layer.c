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
 * @param [in] l The neural network layer to mutate.
 * @param [in] N The number of neurons to add.
 */
void
layer_add_neurons(struct Layer *l, const int N)
{
    const int n_outputs = l->n_outputs + N;
    const int n_weights = n_outputs * l->n_inputs;
    l->weights = realloc(l->weights, sizeof(double) * n_weights);
    l->weight_active = realloc(l->weight_active, sizeof(bool) * n_weights);
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
            const int offset = l->n_inputs * rand_uniform_int(0, l->n_outputs);
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
 * @param [in] l The neural network layer to randomise.
 */
void
layer_weight_rand(struct Layer *l)
{
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
 * @brief Sets n_active to the number of non-zero weights within a layer.
 * @param [in] l The layer to calculate the number of non-zero weights.
 */
void
layer_calc_n_active(struct Layer *l)
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
    l->workspace_size = 0;
}

/**
 * @brief Sets layer parameters to default values.
 * @param [in] args The layer parameters to initialise.
 */
void
layer_args_init(struct ArgsLayer *args)
{
    args->type = CONNECTED;
    args->n_inputs = 0;
    args->n_init = 0;
    args->n_max = 0;
    args->max_neuron_grow = 0;
    args->function = LOGISTIC;
    args->recurrent_function = LOGISTIC;
    args->height = 0;
    args->width = 0;
    args->channels = 0;
    args->n_filters = 0;
    args->size = 0;
    args->stride = 0;
    args->pad = 0;
    args->eta = 0;
    args->eta_min = 0;
    args->momentum = 0;
    args->decay = 0;
    args->probability = 0;
    args->scale = 0;
    args->evolve_weights = false;
    args->evolve_neurons = false;
    args->evolve_functions = false;
    args->evolve_eta = false;
    args->evolve_connect = false;
    args->sgd_weights = false;
    args->next = NULL;
}

/**
 * @brief Creates and returns a copy of specified layer parameters.
 * @param [in] src The layer parameters to be copied.
 */
struct ArgsLayer *
layer_args_copy(const struct ArgsLayer *src)
{
    struct ArgsLayer *new = malloc(sizeof(struct ArgsLayer));
    new->type = src->type;
    new->n_inputs = src->n_inputs;
    new->n_init = src->n_init;
    new->n_max = src->n_max;
    new->max_neuron_grow = src->max_neuron_grow;
    new->function = src->function;
    new->recurrent_function = src->recurrent_function;
    new->height = src->height;
    new->width = src->width;
    new->channels = src->channels;
    new->n_filters = src->n_filters;
    new->size = src->size;
    new->stride = src->stride;
    new->pad = src->pad;
    new->eta = src->eta;
    new->eta_min = src->eta_min;
    new->momentum = src->momentum;
    new->decay = src->decay;
    new->probability = src->probability;
    new->scale = src->scale;
    new->evolve_weights = src->evolve_weights;
    new->evolve_neurons = src->evolve_neurons;
    new->evolve_functions = src->evolve_functions;
    new->evolve_eta = src->evolve_eta;
    new->evolve_connect = src->evolve_connect;
    new->sgd_weights = src->sgd_weights;
    new->next = NULL;
    return new;
}

/**
 * @brief Prints layer input parameters.
 * @param [in] args The layer parameters to print.
 */
static void
layer_args_print_inputs(const struct ArgsLayer *args)
{
    switch (args->type) {
        case CONVOLUTIONAL:
        case MAXPOOL:
        case AVGPOOL:
        case UPSAMPLE:
            if (args->height > 0) {
                printf(", height=%d", args->height);
            }
            if (args->width > 0) {
                printf(", width=%d", args->width);
            }
            if (args->channels > 0) {
                printf(", channels=%d", args->channels);
            }
            if (args->size > 0) {
                printf(", size=%d", args->size);
            }
            if (args->stride > 0) {
                printf(", stride=%d", args->stride);
            }
            if (args->pad > 0) {
                printf(", pad=%d", args->pad);
            }
            break;
        default:
            break;
    }
}

/**
 * @brief Prints layer gradient descent parameters.
 * @param [in] args The layer parameters to print.
 */
static void
layer_args_print_sgd(const struct ArgsLayer *args)
{
    if (args->sgd_weights) {
        printf(", sgd_weights=true");
        printf(", eta=%f", args->eta);
        if (args->evolve_eta) {
            printf(", evolve_eta=true");
            printf(", eta_min=%f", args->eta_min);
        } else {
            printf(", evolve_eta=false");
        }
        printf(", momentum=%f", args->momentum);
        printf(", decay=%f", args->decay);
    }
}

/**
 * @brief Prints layer evolutionary operator parameters.
 * @param [in] args The layer parameters to print.
 */
static void
layer_args_print_evo(const struct ArgsLayer *args)
{
    if (args->evolve_weights) {
        printf(", evolve_weights=true");
    }
    if (args->evolve_functions) {
        printf(", evolve_functions=true");
    }
    if (args->evolve_connect) {
        printf(", evolve_connect=true");
    }
    if (args->evolve_neurons) {
        printf(", evolve_neurons=true");
        printf(", n_max=%d", args->n_max);
        printf(", max_neuron_grow=%d", args->max_neuron_grow);
    }
}

/**
 * @brief Prints layer activation function parameters.
 * @param [in] args The layer parameters to print.
 */
static void
layer_args_print_activation(const struct ArgsLayer *args)
{
    switch (args->type) {
        case AVGPOOL:
        case MAXPOOL:
        case UPSAMPLE:
        case DROPOUT:
        case NOISE:
        case SOFTMAX:
            return;
        default:
            break;
    }
    printf(", activation=%s", neural_activation_string(args->function));
    if (args->type == LSTM) {
        printf(", recurrent_activation=%s",
               neural_activation_string(args->recurrent_function));
    }
}

/**
 * @brief Prints layer parameters.
 * @param [in] args The layer parameters to print.
 */
void
layer_args_print(const struct ArgsLayer *args)
{
    printf("type=%s", layer_type_as_string(args->type));
    layer_args_print_activation(args);
    layer_args_print_inputs(args);
    switch (args->type) {
        case AVGPOOL:
        case MAXPOOL:
        case UPSAMPLE:
            return;
        case CONVOLUTIONAL:
            printf(", n_filters=%d", args->n_filters);
            break;
        case NOISE:
            printf(", probability=%f", args->probability);
            printf(", scale=%f", args->scale);
            break;
        case DROPOUT:
            printf(", probability=%f", args->probability);
            break;
        case SOFTMAX:
            printf(", scale=%f", args->scale);
            return;
        default:
            break;
    }
    if (args->n_init > 0) {
        printf(", n_init=%d", args->n_init);
    }
    layer_args_print_evo(args);
    layer_args_print_sgd(args);
}

/**
 * @brief Frees memory used by a list of layer parameters and points to NULL.
 * @param [in] largs Pointer to the list of layer parameters to free.
 */
void
layer_args_free(struct ArgsLayer **largs)
{
    while (*largs != NULL) {
        struct ArgsLayer *arg = *largs;
        *largs = (*largs)->next;
        free(arg);
    }
}

/**
 * @brief Returns the current output layer arguments.
 * @param [in] head Head of the list of layer parameters.
 * @return Layer parameters pertaining to the current output layer.
 */
struct ArgsLayer *
layer_args_tail(struct ArgsLayer *head)
{
    struct ArgsLayer *tail = head;
    while (tail->next != NULL) {
        tail = tail->next;
    }
    return tail;
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
        case CONNECTED:
        case RECURRENT:
        case LSTM:
        case DROPOUT:
        case NOISE:
        case SOFTMAX:
            return false;
            ;
        default:
            printf("layer_receives_images(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
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
 * @brief Returns a bitstring representing the permissions granted by a layer.
 * @param [in] args Layer initialisation parameters.
 * @return Bitstring representing the layer options.
 */
uint32_t
layer_opt(const struct ArgsLayer *args)
{
    uint32_t lopt = 0;
    if (args->evolve_eta) {
        lopt |= LAYER_EVOLVE_ETA;
    }
    if (args->sgd_weights) {
        lopt |= LAYER_SGD_WEIGHTS;
    }
    if (args->evolve_weights) {
        lopt |= LAYER_EVOLVE_WEIGHTS;
    }
    if (args->evolve_neurons) {
        lopt |= LAYER_EVOLVE_NEURONS;
    }
    if (args->evolve_functions) {
        lopt |= LAYER_EVOLVE_FUNCTIONS;
    }
    if (args->evolve_connect) {
        lopt |= LAYER_EVOLVE_CONNECT;
    }
    return lopt;
}
