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
 * @file neural_layer_args.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Functions operating on neural network arguments/constants.
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
    if (layer_receives_images(args->type)) {
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
        if (args->decay > 0) {
            printf(", decay=%f", args->decay);
        }
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
 * @brief Checks input layer arguments are valid.
 * @param [in] arg Input layer parameters.
 */
static void
layer_args_validate_inputs(struct ArgsLayer *arg)
{
    if (arg->type == DROPOUT || arg->type == NOISE) {
        if (arg->n_inputs < 1) {
            arg->n_inputs = arg->channels * arg->height * arg->width;
        } else if (arg->channels < 1 || arg->height < 1 || arg->width < 1) {
            arg->channels = 1;
            arg->height = 1;
            arg->width = arg->n_inputs;
        }
    }
    if (layer_receives_images(arg->type)) {
        if (arg->channels < 1) {
            printf("Error: input channels < 1\n");
            exit(EXIT_FAILURE);
        }
        if (arg->height < 1) {
            printf("Error: input height < 1\n");
            exit(EXIT_FAILURE);
        }
        if (arg->width < 1) {
            printf("Error: input width < 1\n");
            exit(EXIT_FAILURE);
        }
    } else if (arg->n_inputs < 1) {
        printf("Error: number of inputs < 1\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Checks network layer arguments are valid.
 * @param [in] args List of layer parameters to check.
 */
void
layer_args_validate(struct ArgsLayer *args)
{
    struct ArgsLayer *arg = args;
    if (arg == NULL) {
        printf("Error: empty layer argument list\n");
        exit(EXIT_FAILURE);
    }
    layer_args_validate_inputs(arg);
    do {
        if (arg->evolve_neurons && arg->max_neuron_grow < 1) {
            printf("Error: evolving neurons but max_neuron_grow < 1\n");
            exit(EXIT_FAILURE);
        }
        if (arg->n_max < arg->n_init) {
            arg->n_max = arg->n_init;
        }
        arg = arg->next;
    } while (arg != NULL);
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
 * @brief Returns a bitstring representing the permissions granted by a layer.
 * @param [in] args Layer initialisation parameters.
 * @return Bitstring representing the layer options.
 */
uint32_t
layer_args_opt(const struct ArgsLayer *args)
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
