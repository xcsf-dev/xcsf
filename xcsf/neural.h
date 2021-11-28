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
 * @file neural.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2012--2021.
 * @brief An implementation of a multi-layer perceptron neural network.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ArgsLayer; //!< Forward declaration of layer parameter structure
struct Layer; //!< Forward declaration of layer structure.

/**
 * @brief Double linked list of layers data structure.
 */
struct Llist {
    struct Layer *layer; //!< Pointer to the layer data structure
    struct Llist *prev; //!< Pointer to the previous layer (forward)
    struct Llist *next; //!< Pointer to the next layer (backward)
};

/**
 * @brief Neural network data structure.
 */
struct Net {
    int n_layers; //!< Number of layers (hidden + output)
    int n_inputs; //!< Number of network inputs
    int n_outputs; //!< Number of network outputs
    double *output; //!< Pointer to the network output
    struct Llist *head; //!< Pointer to the head layer (output layer)
    struct Llist *tail; //!< Pointer to the tail layer (first layer)
    bool train; //!< Whether the network is in training mode
};

bool
neural_mutate(const struct Net *net);

const char *
neural_json_export(const struct Net *net, const bool return_weights);

double
neural_output(const struct Net *net, const int IDX);

double *
neural_outputs(const struct Net *net);

double
neural_size(const struct Net *net);

size_t
neural_load(struct Net *net, FILE *fp);

size_t
neural_save(const struct Net *net, FILE *fp);

void
neural_copy(struct Net *dest, const struct Net *src);

void
neural_free(struct Net *net);

void
neural_init(struct Net *net);

void
neural_create(struct Net *net, struct ArgsLayer *arg);

void
neural_insert(struct Net *net, struct Layer *l, const int pos);

void
neural_remove(struct Net *net, const int pos);

void
neural_push(struct Net *net, struct Layer *l);

void
neural_pop(struct Net *net);

void
neural_learn(const struct Net *net, const double *output, const double *input);

void
neural_print(const struct Net *net, const bool print_weights);

void
neural_propagate(struct Net *net, const double *input, const bool train);

void
neural_rand(const struct Net *net);

void
neural_resize(const struct Net *net);
