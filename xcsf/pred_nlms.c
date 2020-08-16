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

#include "pred_nlms.h"
#include "blas.h"
#include "sam.h"
#include "utils.h"

#define ETA_MIN (0.0001) //!< Minimum gradient descent rate
#define N_MU (1) //!< Number of self-adaptive mutation rates
static const int MU_TYPE[N_MU] = {
    SAM_LOG_NORMAL //!< Rate of gradient descent mutation
}; //<! Self-adaptation method

void
pred_nlms_init(const struct XCSF *xcsf, struct CL *c)
{
    struct PRED_NLMS *pred = malloc(sizeof(struct PRED_NLMS));
    c->pred = pred;
    // set the length of weights per predicted variable
    if (xcsf->PRED_TYPE == PRED_TYPE_NLMS_QUADRATIC) {
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
    if (xcsf->PRED_EVOLVE_ETA) {
        sam_init(pred->mu, N_MU, MU_TYPE);
        pred->eta = rand_uniform(ETA_MIN, xcsf->PRED_ETA);
    } else {
        memset(pred->mu, 0, sizeof(double) * N_MU);
        pred->eta = xcsf->PRED_ETA;
    }
    // initialise temporary storage for weight updating
    pred->tmp_input = malloc(sizeof(double) * pred->n);
}

void
pred_nlms_copy(const struct XCSF *xcsf, struct CL *dest, const struct CL *src)
{
    pred_nlms_init(xcsf, dest);
    struct PRED_NLMS *dest_pred = dest->pred;
    const struct PRED_NLMS *src_pred = src->pred;
    memcpy(dest_pred->weights, src_pred->weights,
           sizeof(double) * src_pred->n_weights);
    memcpy(dest_pred->mu, src_pred->mu, sizeof(double) * N_MU);
    dest_pred->eta = src_pred->eta;
}

void
pred_nlms_free(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    struct PRED_NLMS *pred = c->pred;
    free(pred->weights);
    free(pred->tmp_input);
    free(pred->mu);
    free(pred);
}

void
pred_nlms_update(const struct XCSF *xcsf, const struct CL *c, const double *x,
                 const double *y)
{
    const struct PRED_NLMS *pred = c->pred;
    int n = pred->n;
    // normalise update
    double norm = xcsf->PRED_X0 * xcsf->PRED_X0;
    norm += blas_dot(xcsf->x_dim, x, 1, x, 1);
    // update weights using the error
    for (int i = 0; i < xcsf->y_dim; ++i) {
        double error = y[i] - c->prediction[i];
        double correction = (pred->eta * error) / norm;
        blas_axpy(n, correction, pred->tmp_input, 1, &pred->weights[i * n], 1);
    }
}

void
pred_nlms_compute(const struct XCSF *xcsf, const struct CL *c, const double *x)
{
    const struct PRED_NLMS *pred = c->pred;
    int n = pred->n;
    pred_transform_input(xcsf, x, pred->tmp_input);
    for (int i = 0; i < xcsf->y_dim; ++i) {
        c->prediction[i] =
            blas_dot(n, &pred->weights[i * n], 1, pred->tmp_input, 1);
    }
}

void
pred_nlms_print(const struct XCSF *xcsf, const struct CL *c)
{
    const struct PRED_NLMS *pred = c->pred;
    int n = pred->n;
    printf("eta: %.5f, weights: ", pred->eta);
    for (int i = 0; i < xcsf->y_dim; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f, ", pred->weights[i * n + j]);
        }
        printf("\n");
    }
}

_Bool
pred_nlms_crossover(const struct XCSF *xcsf, const struct CL *c1,
                    const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
pred_nlms_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    if (xcsf->PRED_EVOLVE_ETA) {
        struct PRED_NLMS *pred = c->pred;
        sam_adapt(pred->mu, N_MU, MU_TYPE);
        double orig = pred->eta;
        pred->eta += rand_normal(0, pred->mu[0]);
        pred->eta = clamp(pred->eta, ETA_MIN, xcsf->PRED_ETA);
        if (orig != pred->eta) {
            return true;
        }
    }
    return false;
}

double
pred_nlms_size(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    const struct PRED_NLMS *pred = c->pred;
    return pred->n_weights;
}

size_t
pred_nlms_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    (void) xcsf;
    const struct PRED_NLMS *pred = c->pred;
    size_t s = 0;
    s += fwrite(&pred->n, sizeof(int), 1, fp);
    s += fwrite(&pred->n_weights, sizeof(int), 1, fp);
    s += fwrite(pred->weights, sizeof(double), pred->n_weights, fp);
    s += fwrite(pred->mu, sizeof(double), N_MU, fp);
    s += fwrite(&pred->eta, sizeof(double), 1, fp);
    return s;
}

size_t
pred_nlms_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    pred_nlms_init(xcsf, c);
    struct PRED_NLMS *pred = c->pred;
    size_t s = 0;
    s += fread(&pred->n, sizeof(int), 1, fp);
    s += fread(&pred->n_weights, sizeof(int), 1, fp);
    s += fread(pred->weights, sizeof(double), pred->n_weights, fp);
    s += fread(pred->mu, sizeof(double), N_MU, fp);
    s += fread(&pred->eta, sizeof(double), 1, fp);
    return s;
}
