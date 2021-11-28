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
 * @file pred_rls.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Recursive least mean squares prediction functions.
 */

#pragma once

#include "prediction.h"
#include "xcsf.h"

/**
 * @brief Recursive least mean squares prediction data structure.
 */
struct PredRLS {
    int n; //!< Number of weights for each predicted variable
    int n_weights; //!< Total number of weights
    double *weights; //!< Weights used to compute prediction
    double *matrix; //!< Gain matrix used to update weights
    double *tmp_input; //!< Temporary storage for updating weights
    double *tmp_vec; //!< Temporary storage for updating weights
    double *tmp_matrix1; //!< Temporary storage for updating gain matrix
    double *tmp_matrix2; //!< Temporary storage for updating gain matrix
};

bool
pred_rls_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2);

bool
pred_rls_mutate(const struct XCSF *xcsf, const struct Cl *c);

double
pred_rls_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
pred_rls_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

size_t
pred_rls_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

void
pred_rls_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x);

void
pred_rls_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src);

void
pred_rls_free(const struct XCSF *xcsf, const struct Cl *c);

void
pred_rls_init(const struct XCSF *xcsf, struct Cl *c);

void
pred_rls_print(const struct XCSF *xcsf, const struct Cl *c);

void
pred_rls_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                const double *y);

const char *
pred_rls_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief Recursive least mean squares prediction implemented functions.
 */
static struct PredVtbl const pred_rls_vtbl = {
    &pred_rls_crossover, &pred_rls_mutate, &pred_rls_compute,
    &pred_rls_copy,      &pred_rls_free,   &pred_rls_init,
    &pred_rls_print,     &pred_rls_update, &pred_rls_size,
    &pred_rls_save,      &pred_rls_load,   &pred_rls_json_export
};
