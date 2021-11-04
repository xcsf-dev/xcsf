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
 * @file neural_layer_maxpool.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief An implementation of a 2D maxpooling layer.
 */

#pragma once

#include "neural_layer.h"

void
neural_layer_maxpool_init(struct Layer *l, const struct ArgsLayer *args);

struct Layer *
neural_layer_maxpool_copy(const struct Layer *src);

void
neural_layer_maxpool_rand(struct Layer *l);

void
neural_layer_maxpool_forward(const struct Layer *l, const struct Net *net,
                             const double *input);

void
neural_layer_maxpool_backward(const struct Layer *l, const struct Net *net,
                              const double *input, double *delta);

void
neural_layer_maxpool_update(const struct Layer *l);

void
neural_layer_maxpool_print(const struct Layer *l, const bool print_weights);

bool
neural_layer_maxpool_mutate(struct Layer *l);

void
neural_layer_maxpool_free(const struct Layer *l);

double *
neural_layer_maxpool_output(const struct Layer *l);

size_t
neural_layer_maxpool_save(const struct Layer *l, FILE *fp);

size_t
neural_layer_maxpool_load(struct Layer *l, FILE *fp);

void
neural_layer_maxpool_resize(struct Layer *l, const struct Layer *prev);

const char *
neural_layer_maxpool_json(const struct Layer *l, const bool return_weights);

/**
 * @brief Neural maxpooling layer implemented functions.
 */
static struct LayerVtbl const layer_maxpool_vtbl = {
    &neural_layer_maxpool_init,     &neural_layer_maxpool_mutate,
    &neural_layer_maxpool_resize,   &neural_layer_maxpool_copy,
    &neural_layer_maxpool_free,     &neural_layer_maxpool_rand,
    &neural_layer_maxpool_print,    &neural_layer_maxpool_update,
    &neural_layer_maxpool_backward, &neural_layer_maxpool_forward,
    &neural_layer_maxpool_output,   &neural_layer_maxpool_save,
    &neural_layer_maxpool_load,     &neural_layer_maxpool_json
};
