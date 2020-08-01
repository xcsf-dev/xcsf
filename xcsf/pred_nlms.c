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
 * @file pred_nlms.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Normalised least mean squares prediction functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"
#include "utils.h"
#include "blas.h"
#include "sam.h"
#include "cl.h"
#include "prediction.h"
#include "pred_nlms.h"

#define N_MU 1 //!< Number of self-adaptive mutation rates
#define ETA_MIN 0.0001 //!< Minimum gradient descent rate

void pred_nlms_init(const XCSF *xcsf, CL *c)
{
    PRED_NLMS *pred = malloc(sizeof(PRED_NLMS));
    c->pred = pred;
    // set the length of weights per predicted variable
    if(xcsf->PRED_TYPE == PRED_TYPE_NLMS_QUADRATIC) {
        // offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
        pred->n = 1 + 2 * xcsf->x_dim + xcsf->x_dim * (xcsf->x_dim - 1) / 2;
    } else {
        pred->n = xcsf->x_dim + 1;
    }
    // initialise weights
    pred->n_weights = pred->n * xcsf->y_dim;
    pred->weights = calloc(pred->n_weights, sizeof(double));
    blas_fill(xcsf->y_dim, xcsf->PRED_X0, pred->weights, pred->n);
    // initialise learning rate
    pred->mu = malloc(sizeof(double) * N_MU);
    if(xcsf->PRED_EVOLVE_ETA) {
        sam_init(xcsf, pred->mu, N_MU);
        pred->eta = rand_uniform(ETA_MIN, xcsf->PRED_ETA);
    } else {
        memset(pred->mu, 0, sizeof(double) * N_MU);
        pred->eta = xcsf->PRED_ETA;
    }
    // initialise temporary storage for weight updating
    pred->tmp_input = malloc(sizeof(double) * pred->n);
}

void pred_nlms_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    pred_nlms_init(xcsf, dest);
    PRED_NLMS *dest_pred = dest->pred;
    const PRED_NLMS *src_pred = src->pred;
    memcpy(dest_pred->weights, src_pred->weights, sizeof(double) * src_pred->n_weights);
    memcpy(dest_pred->mu, src_pred->mu, sizeof(double) * N_MU);
    dest_pred->eta = src_pred->eta;
}

void pred_nlms_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    PRED_NLMS *pred = c->pred;
    free(pred->weights);
    free(pred->tmp_input);
    free(pred->mu);
    free(pred);
}

void pred_nlms_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    const PRED_NLMS *pred = c->pred;
    int n = pred->n;
    // normalise update
    double norm = xcsf->PRED_X0 * xcsf->PRED_X0;
    norm += blas_dot(xcsf->x_dim, x, 1, x, 1);
    // update weights using the error
    for(int var = 0; var < xcsf->y_dim; ++var) {
        double error = y[var] - c->prediction[var];
        double correction = (pred->eta * error) / norm;
        blas_axpy(n, correction, pred->tmp_input, 1, &pred->weights[var * n], 1);
    }
}

void pred_nlms_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    const PRED_NLMS *pred = c->pred;
    int n = pred->n;
    pred_transform_input(xcsf, x, pred->tmp_input);
    for(int var = 0; var < xcsf->y_dim; ++var) {
        c->prediction[var] = blas_dot(n, &pred->weights[var * n], 1, pred->tmp_input, 1);
    }
}

void pred_nlms_print(const XCSF *xcsf, const CL *c)
{
    const PRED_NLMS *pred = c->pred;
    int n = pred->n;
    printf("eta: %.5f, weights: ", pred->eta);
    for(int var = 0; var < xcsf->y_dim; ++var) {
        for(int i = 0; i < n; ++i) {
            printf("%f, ", pred->weights[var * n + i]);
        }
        printf("\n");
    }
}

_Bool pred_nlms_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}

_Bool pred_nlms_mutate(const XCSF *xcsf, const CL *c)
{
    if(xcsf->PRED_EVOLVE_ETA) {
        PRED_NLMS *pred = c->pred;
        sam_adapt(xcsf, pred->mu, N_MU);
        double orig = pred->eta;
        pred->eta += rand_normal(0, pred->mu[0]);
        pred->eta = clamp(ETA_MIN, xcsf->PRED_ETA, pred->eta);
        if(orig != pred->eta) {
            return true;
        }
    }
    return false;
}

int pred_nlms_size(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const PRED_NLMS *pred = c->pred;
    return pred->n_weights;
}

size_t pred_nlms_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf;
    const PRED_NLMS *pred = c->pred;
    size_t s = 0;
    s += fwrite(&pred->n, sizeof(int), 1, fp);
    s += fwrite(&pred->n_weights, sizeof(int), 1, fp);
    s += fwrite(pred->weights, sizeof(double), pred->n_weights, fp);
    s += fwrite(pred->mu, sizeof(double), N_MU, fp);
    s += fwrite(&pred->eta, sizeof(double), 1, fp);
    return s;
}

size_t pred_nlms_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    pred_nlms_init(xcsf, c);
    PRED_NLMS *pred = c->pred;
    size_t s = 0;
    s += fread(&pred->n, sizeof(int), 1, fp);
    s += fread(&pred->n_weights, sizeof(int), 1, fp);
    s += fread(pred->weights, sizeof(double), pred->n_weights, fp);
    s += fread(pred->mu, sizeof(double), N_MU, fp);
    s += fread(&pred->eta, sizeof(double), 1, fp);
    return s;
}
