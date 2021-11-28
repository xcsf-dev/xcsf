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
 * @file neural_layer_recurrent.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief An implementation of a recurrent layer of perceptrons.
 */

#pragma once

#include "neural_layer.h"

void
neural_layer_recurrent_init(struct Layer *l, const struct ArgsLayer *args);

struct Layer *
neural_layer_recurrent_copy(const struct Layer *src);

void
neural_layer_recurrent_rand(struct Layer *l);

void
neural_layer_recurrent_forward(const struct Layer *l, const struct Net *net,
                               const double *input);

void
neural_layer_recurrent_backward(const struct Layer *l, const struct Net *net,
                                const double *input, double *delta);

void
neural_layer_recurrent_update(const struct Layer *l);

void
neural_layer_recurrent_print(const struct Layer *l, const bool print_weights);

bool
neural_layer_recurrent_mutate(struct Layer *l);

void
neural_layer_recurrent_free(const struct Layer *l);

double *
neural_layer_recurrent_output(const struct Layer *l);

size_t
neural_layer_recurrent_save(const struct Layer *l, FILE *fp);

size_t
neural_layer_recurrent_load(struct Layer *l, FILE *fp);

void
neural_layer_recurrent_resize(struct Layer *l, const struct Layer *prev);

const char *
neural_layer_recurrent_json_export(const struct Layer *l,
                                   const bool return_weights);

/**
 * @brief Neural recurrent layer implemented functions.
 */
static struct LayerVtbl const layer_recurrent_vtbl = {
    &neural_layer_recurrent_init,     &neural_layer_recurrent_mutate,
    &neural_layer_recurrent_resize,   &neural_layer_recurrent_copy,
    &neural_layer_recurrent_free,     &neural_layer_recurrent_rand,
    &neural_layer_recurrent_print,    &neural_layer_recurrent_update,
    &neural_layer_recurrent_backward, &neural_layer_recurrent_forward,
    &neural_layer_recurrent_output,   &neural_layer_recurrent_save,
    &neural_layer_recurrent_load,     &neural_layer_recurrent_json_export
};
