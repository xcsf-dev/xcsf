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
 * @file cond_ternary.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2021.
 * @brief Ternary condition functions.
 */

#pragma once

#include "condition.h"
#include "xcsf.h"

/**
 * @brief Ternary condition data structure.
 */
struct CondTernary {
    char *string; //!< Ternary bitstring
    int length; //!< Length of the bitstring
    double *mu; //!< Mutation rates
    char *tmp_input; //!< Temporary storage for float conversion
};

bool
cond_ternary_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                       const struct Cl *c2);

bool
cond_ternary_general(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2);

bool
cond_ternary_match(const struct XCSF *xcsf, const struct Cl *c,
                   const double *x);

bool
cond_ternary_mutate(const struct XCSF *xcsf, const struct Cl *c);

void
cond_ternary_copy(const struct XCSF *xcsf, struct Cl *dest,
                  const struct Cl *src);

void
cond_ternary_cover(const struct XCSF *xcsf, const struct Cl *c,
                   const double *x);

void
cond_ternary_free(const struct XCSF *xcsf, const struct Cl *c);

void
cond_ternary_init(const struct XCSF *xcsf, struct Cl *c);

void
cond_ternary_print(const struct XCSF *xcsf, const struct Cl *c);

void
cond_ternary_update(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x, const double *y);

double
cond_ternary_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
cond_ternary_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

size_t
cond_ternary_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

const char *
cond_ternary_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief Ternary condition implemented functions.
 */
static struct CondVtbl const cond_ternary_vtbl = {
    &cond_ternary_crossover, &cond_ternary_general,    &cond_ternary_match,
    &cond_ternary_mutate,    &cond_ternary_copy,       &cond_ternary_cover,
    &cond_ternary_free,      &cond_ternary_init,       &cond_ternary_print,
    &cond_ternary_update,    &cond_ternary_size,       &cond_ternary_save,
    &cond_ternary_load,      &cond_ternary_json_export
};
