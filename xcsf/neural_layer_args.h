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
 * @file neural_layer_args.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020--2021.
 * @brief Functions operating on neural network arguments/constants.
 */

/**
 * @brief Parameters for initialising a neural network layer.
 */
struct ArgsLayer {
    int type; //!< Layer type: CONNECTED, DROPOUT, etc.
    int n_inputs; //!< Number of inputs
    int n_init; //!< Initial number of units / neurons / filters
    int n_max; //!< Maximum number of units / neurons
    int max_neuron_grow; //!< Maximum number neurons to add per mutation event
    int function; //!< Activation function
    int recurrent_function; //!< Recurrent activation function
    int height; //!< Pool, Conv, and Upsample
    int width; //!< Pool, Conv, and Upsample
    int channels; //!< Pool, Conv, and Upsample
    int size; //!< Pool and Conv
    int stride; //!< Pool, Conv, and Upsample
    int pad; //!< Pool and Conv
    double eta; //!< Gradient descent rate
    double eta_min; //!< Current gradient descent rate
    double momentum; //!< Momentum for gradient descent
    double decay; //!< Weight decay for gradient descent
    double probability; //!< Usage depends on layer implementation
    double scale; //!< Usage depends on layer implementation
    _Bool evolve_weights; //!< Ability to evolve weights
    _Bool evolve_neurons; //!< Ability to evolve number of units
    _Bool evolve_functions; //!< Ability to evolve activation function
    _Bool evolve_eta; //!< Ability to evolve gradient descent rate
    _Bool evolve_connect; //!< Ability to evolve weight connectivity
    _Bool sgd_weights; //!< Ability to update weights with gradient descent
    struct ArgsLayer *next; //!< Next layer parameters
};

void
layer_args_init(struct ArgsLayer *args);

struct ArgsLayer *
layer_args_copy(const struct ArgsLayer *src);

struct ArgsLayer *
layer_args_tail(struct ArgsLayer *head);

const char *
layer_args_json_export(struct ArgsLayer *args);

void
layer_args_free(struct ArgsLayer **largs);

void
layer_args_validate(struct ArgsLayer *args);

uint32_t
layer_args_opt(const struct ArgsLayer *args);

size_t
layer_args_save(const struct ArgsLayer *args, FILE *fp);

size_t
layer_args_load(struct ArgsLayer **largs, FILE *fp);
