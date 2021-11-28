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
 * @file cond_neural.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief Multi-layer perceptron neural network condition functions.
 */

#pragma once

#include "condition.h"
#include "neural.h"
#include "neural_activations.h"
#include "neural_layer.h"
#include "xcsf.h"

/**
 * @brief Multi-layer perceptron neural network condition data structure.
 */
struct CondNeural {
    struct Net net; //!< Neural network
};

bool
cond_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2);

bool
cond_neural_general(const struct XCSF *xcsf, const struct Cl *c1,
                    const struct Cl *c2);

bool
cond_neural_match(const struct XCSF *xcsf, const struct Cl *c, const double *x);

bool
cond_neural_mutate(const struct XCSF *xcsf, const struct Cl *c);

void
cond_neural_copy(const struct XCSF *xcsf, struct Cl *dest,
                 const struct Cl *src);

void
cond_neural_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x);

void
cond_neural_free(const struct XCSF *xcsf, const struct Cl *c);

void
cond_neural_init(const struct XCSF *xcsf, struct Cl *c);

void
cond_neural_print(const struct XCSF *xcsf, const struct Cl *c);

void
cond_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const double *y);

double
cond_neural_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
cond_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

size_t
cond_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

int
cond_neural_neurons(const struct XCSF *xcsf, const struct Cl *c, int layer);

int
cond_neural_layers(const struct XCSF *xcsf, const struct Cl *c);

int
cond_neural_connections(const struct XCSF *xcsf, const struct Cl *c, int layer);

const char *
cond_neural_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief Multi-layer perceptron neural network condition implemented functions.
 */
static struct CondVtbl const cond_neural_vtbl = {
    &cond_neural_crossover, &cond_neural_general,    &cond_neural_match,
    &cond_neural_mutate,    &cond_neural_copy,       &cond_neural_cover,
    &cond_neural_free,      &cond_neural_init,       &cond_neural_print,
    &cond_neural_update,    &cond_neural_size,       &cond_neural_save,
    &cond_neural_load,      &cond_neural_json_export
};
