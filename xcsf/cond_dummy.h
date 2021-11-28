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
 * @file cond_dummy.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019-2021.
 * @brief Always-matching dummy condition functions.
 */

#pragma once

#include "condition.h"
#include "xcsf.h"

bool
cond_dummy_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2);

bool
cond_dummy_general(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2);

bool
cond_dummy_match(const struct XCSF *xcsf, const struct Cl *c, const double *x);

bool
cond_dummy_mutate(const struct XCSF *xcsf, const struct Cl *c);

void
cond_dummy_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src);

void
cond_dummy_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x);

void
cond_dummy_free(const struct XCSF *xcsf, const struct Cl *c);

void
cond_dummy_init(const struct XCSF *xcsf, struct Cl *c);

void
cond_dummy_print(const struct XCSF *xcsf, const struct Cl *c);

void
cond_dummy_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                  const double *y);

double
cond_dummy_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
cond_dummy_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

size_t
cond_dummy_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

const char *
cond_dummy_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief Dummy condition implemented functions.
 */
static struct CondVtbl const cond_dummy_vtbl = {
    &cond_dummy_crossover, &cond_dummy_general,    &cond_dummy_match,
    &cond_dummy_mutate,    &cond_dummy_copy,       &cond_dummy_cover,
    &cond_dummy_free,      &cond_dummy_init,       &cond_dummy_print,
    &cond_dummy_update,    &cond_dummy_size,       &cond_dummy_save,
    &cond_dummy_load,      &cond_dummy_json_export
};
