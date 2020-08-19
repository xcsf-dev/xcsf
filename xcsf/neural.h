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
 * @date 2012--2020.
 * @brief An implementation of a multi-layer perceptron neural network.
 */

#pragma once

#include "xcsf.h"

/**
 * @brief Double linked list of layers data structure.
 */
typedef struct LLIST {
    struct LAYER *layer; //!< Pointer to the layer data structure
    struct LLIST *prev; //!< Pointer to the previous layer (forward)
    struct LLIST *next; //!< Pointer to the next layer (backward)
} LLIST;

/**
 * @brief Neural network data structure.
 */
typedef struct NET {
    int n_layers; //!< Number of layers (hidden + output)
    int n_inputs; //!< Number of network inputs
    int n_outputs; //!< Number of network outputs
    double *output; //!< Pointer to the network output
    struct LLIST *head; //!< Pointer to the head layer (output layer)
    struct LLIST *tail; //!< Pointer to the tail layer (first layer)
} NET;

_Bool
neural_mutate(const struct XCSF *xcsf, const struct NET *net);

double
neural_output(const struct XCSF *xcsf, const struct NET *net, const int IDX);

double
neural_size(const struct XCSF *xcsf, const struct NET *net);

size_t
neural_load(const struct XCSF *xcsf, struct NET *net, FILE *fp);

size_t
neural_save(const struct XCSF *xcsf, const struct NET *net, FILE *fp);

void
neural_copy(const struct XCSF *xcsf, struct NET *dest, const struct NET *src);

void
neural_free(const struct XCSF *xcsf, struct NET *net);

void
neural_init(const struct XCSF *xcsf, struct NET *net);

void
neural_layer_insert(const struct XCSF *xcsf, struct NET *net, struct LAYER *l,
                    const int pos);

void
neural_layer_remove(const struct XCSF *xcsf, struct NET *net, const int pos);

void
neural_learn(const struct XCSF *xcsf, const struct NET *net,
             const double *output, const double *input);

void
neural_print(const struct XCSF *xcsf, const struct NET *net,
             const _Bool print_weights);

void
neural_propagate(const struct XCSF *xcsf, const struct NET *net,
                 const double *input);

void
neural_rand(const struct XCSF *xcsf, const struct NET *net);

void
neural_resize(const struct XCSF *xcsf, const struct NET *net);
