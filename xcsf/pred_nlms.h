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
 * @file pred_nlms.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Normalised least mean squares prediction functions.
 */

#pragma once

#include "prediction.h"
#include "xcsf.h"

/**
 * @brief Normalised least mean squares prediction data structure.
 */
struct PredNLMS {
    int n; //!< Number of weights for each predicted variable
    int n_weights; //!< Total number of weights
    double *weights; //!< Weights used to compute prediction
    double *mu; //!< Mutation rates
    double eta; //!< Gradient descent rate
    double *tmp_input; //!< Temporary storage for updating weights
};

bool
pred_nlms_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                    const struct Cl *c2);

bool
pred_nlms_mutate(const struct XCSF *xcsf, const struct Cl *c);

double
pred_nlms_size(const struct XCSF *xcsf, const struct Cl *c);

size_t
pred_nlms_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp);

size_t
pred_nlms_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp);

void
pred_nlms_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x);

void
pred_nlms_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src);

void
pred_nlms_free(const struct XCSF *xcsf, const struct Cl *c);

void
pred_nlms_init(const struct XCSF *xcsf, struct Cl *c);

void
pred_nlms_print(const struct XCSF *xcsf, const struct Cl *c);

void
pred_nlms_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                 const double *y);

const char *
pred_nlms_json_export(const struct XCSF *xcsf, const struct Cl *c);

/**
 * @brief Normalised least mean squares prediction implemented functions.
 */
static struct PredVtbl const pred_nlms_vtbl = {
    &pred_nlms_crossover, &pred_nlms_mutate, &pred_nlms_compute,
    &pred_nlms_copy,      &pred_nlms_free,   &pred_nlms_init,
    &pred_nlms_print,     &pred_nlms_update, &pred_nlms_size,
    &pred_nlms_save,      &pred_nlms_load,   &pred_nlms_json_export
};
