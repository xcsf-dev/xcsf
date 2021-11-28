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
 * @file pred_constant.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Piece-wise constant prediction functions.
 */

#pragma once

#include "prediction.h"
#include "xcsf.h"

bool
pred_constant_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                        const struct Cl *c2);

bool
pred_constant_mutate(const struct XCSF *xcsf, const struct Cl *c);

double
pred_constant_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
pred_constant_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

size_t
pred_constant_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

void
pred_constant_compute(const struct XCSF *xcsf, const struct Cl *c,
                      const double *x);

void
pred_constant_copy(const struct XCSF *xcsf, struct Cl *dest,
                   const struct Cl *src);

void
pred_constant_free(const struct XCSF *xcsf, const struct Cl *c);

void
pred_constant_init(const struct XCSF *xcsf, struct Cl *c);

void
pred_constant_print(const struct XCSF *xcsf, const struct Cl *c);

void
pred_constant_update(const struct XCSF *xcsf, const struct Cl *c,
                     const double *x, const double *y);

const char *
pred_constant_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief Constant prediction implemented functions.
 */
static struct PredVtbl const pred_constant_vtbl = {
    &pred_constant_crossover, &pred_constant_mutate, &pred_constant_compute,
    &pred_constant_copy,      &pred_constant_free,   &pred_constant_init,
    &pred_constant_print,     &pred_constant_update, &pred_constant_size,
    &pred_constant_save,      &pred_constant_load,   &pred_constant_json_export
};
