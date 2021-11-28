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
 * @file cond_dgp.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief Dynamical GP graph condition functions.
 */

#pragma once

#include "condition.h"
#include "dgp.h"
#include "xcsf.h"

/**
 * @brief Dynamical GP graph condition data structure.
 */
struct CondDGP {
    struct Graph dgp; //!< DGP graph
};

bool
cond_dgp_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2);

bool
cond_dgp_general(const struct XCSF *xcsf, const struct Cl *c1,
                 const struct Cl *c2);

bool
cond_dgp_match(const struct XCSF *xcsf, const struct Cl *c, const double *x);

bool
cond_dgp_mutate(const struct XCSF *xcsf, const struct Cl *c);

void
cond_dgp_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src);

void
cond_dgp_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x);

void
cond_dgp_free(const struct XCSF *xcsf, const struct Cl *c);

void
cond_dgp_init(const struct XCSF *xcsf, struct Cl *c);

void
cond_dgp_print(const struct XCSF *xcsf, const struct Cl *c);

void
cond_dgp_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                const double *y);

double
cond_dgp_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
cond_dgp_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

size_t
cond_dgp_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

const char *
cond_dgp_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief Dynamical GP graph condition implemented functions.
 */
static struct CondVtbl const cond_dgp_vtbl = {
    &cond_dgp_crossover, &cond_dgp_general,    &cond_dgp_match,
    &cond_dgp_mutate,    &cond_dgp_copy,       &cond_dgp_cover,
    &cond_dgp_free,      &cond_dgp_init,       &cond_dgp_print,
    &cond_dgp_update,    &cond_dgp_size,       &cond_dgp_save,
    &cond_dgp_load,      &cond_dgp_json_export
};
