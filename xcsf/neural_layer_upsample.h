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
 * @file neural_layer_upsample.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief An implementation of a 2D upsampling layer.
 */

#pragma once

#include "neural_layer.h"

void
neural_layer_upsample_init(struct Layer *l, const struct ArgsLayer *args);

struct Layer *
neural_layer_upsample_copy(const struct Layer *src);

void
neural_layer_upsample_rand(struct Layer *l);

void
neural_layer_upsample_forward(const struct Layer *l, const struct Net *net,
                              const double *input);

void
neural_layer_upsample_backward(const struct Layer *l, const struct Net *net,
                               const double *input, double *delta);

void
neural_layer_upsample_update(const struct Layer *l);

void
neural_layer_upsample_print(const struct Layer *l, const bool print_weights);

bool
neural_layer_upsample_mutate(struct Layer *l);

void
neural_layer_upsample_free(const struct Layer *l);

double *
neural_layer_upsample_output(const struct Layer *l);

size_t
neural_layer_upsample_save(const struct Layer *l, FILE *fp);

size_t
neural_layer_upsample_load(struct Layer *l, FILE *fp);

void
neural_layer_upsample_resize(struct Layer *l, const struct Layer *prev);

const char *
neural_layer_upsample_json_export(const struct Layer *l,
                                  const bool return_weights);

/**
 * @brief Neural upsampling layer implemented functions.
 */
static struct LayerVtbl const layer_upsample_vtbl = {
    &neural_layer_upsample_init,     &neural_layer_upsample_mutate,
    &neural_layer_upsample_resize,   &neural_layer_upsample_copy,
    &neural_layer_upsample_free,     &neural_layer_upsample_rand,
    &neural_layer_upsample_print,    &neural_layer_upsample_update,
    &neural_layer_upsample_backward, &neural_layer_upsample_forward,
    &neural_layer_upsample_output,   &neural_layer_upsample_save,
    &neural_layer_upsample_load,     &neural_layer_upsample_json_export
};
