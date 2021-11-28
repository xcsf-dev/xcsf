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
 * @file act_neural.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020--2021.
 * @brief Neural network action functions.
 */

#pragma once

#include "action.h"
#include "neural.h"
#include "neural_activations.h"
#include "neural_layer.h"
#include "xcsf.h"

/**
 * @brief Neural network action data structure.
 */
struct ActNeural {
    struct Net net; //!< Neural network
};

bool
act_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2);

bool
act_neural_general(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2);

bool
act_neural_mutate(const struct XCSF *xcsf, const struct Cl *c);

int
act_neural_compute(const struct XCSF *xcsf, const struct Cl *c,
                   const double *x);

void
act_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src);

void
act_neural_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                 const int action);

void
act_neural_free(const struct XCSF *xcsf, const struct Cl *c);

void
act_neural_init(const struct XCSF *xcsf, struct Cl *c);

void
act_neural_print(const struct XCSF *xcsf, const struct Cl *c);

void
act_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                  const double *y);

size_t
act_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

size_t
act_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

const char *
act_neural_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief neural action implemented functions.
 */
static struct ActVtbl const act_neural_vtbl = {
    &act_neural_general,    &act_neural_crossover, &act_neural_mutate,
    &act_neural_compute,    &act_neural_copy,      &act_neural_cover,
    &act_neural_free,       &act_neural_init,      &act_neural_print,
    &act_neural_update,     &act_neural_save,      &act_neural_load,
    &act_neural_json_export
};
