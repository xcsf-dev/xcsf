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
#define ETA_MAX 0.1 //!< Maximum gradient descent rate
#define ETA_MIN 0.0001 //!< Minimum gradient descent rate

/**
 * @brief Normalised least mean squares prediction data structure.
 */ 
typedef struct PRED_NLMS {
    int n; //!< Number of weights for each predicted variable
    int n_weights; //!< Total number of weights
    double *weights; //!< Weights used to compute prediction
    double mu[N_MU]; //!< Mutation rates
    double eta; //!< Gradient descent rate
} PRED_NLMS;

void pred_nlms_init(const XCSF *xcsf, CL *c)
{
    PRED_NLMS *pred = malloc(sizeof(PRED_NLMS));
    c->pred = pred;
    // set the length of weights per predicted variable
    if(xcsf->PRED_TYPE == PRED_TYPE_NLMS_QUADRATIC) {
        // offset(1) + n linear + n quadratic + n*(n-1)/2 mixed terms
        pred->n = 1 + 2 * xcsf->x_dim + xcsf->x_dim * (xcsf->x_dim - 1) / 2;
    }
    else {
        pred->n = xcsf->x_dim + 1;
    }
    // initialise weights
    pred->n_weights = pred->n * xcsf->y_dim;
    pred->weights = calloc(pred->n_weights, sizeof(double));
    blas_fill(xcsf->y_dim, xcsf->PRED_X0, pred->weights, pred->n);
    // initialise learning rate
    if(xcsf->PRED_EVOLVE_ETA) {
        sam_init(xcsf, pred->mu, N_MU);
        pred->eta = rand_uniform(ETA_MIN, ETA_MAX);
    }
    else {
        memset(pred->mu, 0, sizeof(double) * N_MU);
        pred->eta = xcsf->PRED_ETA;
    }
}

void pred_nlms_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    pred_nlms_init(xcsf, to);
    PRED_NLMS *to_pred = to->pred;
    const PRED_NLMS *from_pred = from->pred;
    memcpy(to_pred->weights, from_pred->weights, from_pred->n_weights * sizeof(double));
    memcpy(to_pred->mu, from_pred->mu, N_MU);
    to_pred->eta = from_pred->eta;
}

void pred_nlms_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    PRED_NLMS *pred = c->pred;
    free(pred->weights);
    free(pred);
}

void pred_nlms_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    const PRED_NLMS *pred = c->pred;
    int n = pred->n;
    double norm = xcsf->PRED_X0 * xcsf->PRED_X0;
    for(int i = 0; i < xcsf->x_dim; i++) {
        norm += x[i] * x[i];
    }      
    // update weights using the error
    for(int var = 0; var < xcsf->y_dim; var++) {
        double error = y[var] - c->prediction[var];
        double correction = (pred->eta * error) / norm;
        // update first coefficient
        pred->weights[var*n] += xcsf->PRED_X0 * correction;
        int idx = 1;
        // update linear coefficients
        for(int i = 0; i < xcsf->x_dim; i++) {
            pred->weights[var*n+idx] += correction * x[i];
            idx++;
        }
        if(xcsf->PRED_TYPE == PRED_TYPE_NLMS_QUADRATIC) {
            // update quadratic coefficients
            for(int i = 0; i < xcsf->x_dim; i++) {
                for(int j = i; j < xcsf->x_dim; j++) {
                    pred->weights[var*n+idx] += correction * x[i] * x[j];
                    idx++;
                }
            }
        }
    }
}

void pred_nlms_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    const PRED_NLMS *pred = c->pred;
    int n = pred->n;
    for(int var = 0; var < xcsf->y_dim; var++) {
        // first coefficient is offset
        double pre = xcsf->PRED_X0 * pred->weights[var*n];
        int idx = 1;
        // multiply linear coefficients with the prediction input
        for(int i = 0; i < xcsf->x_dim; i++) {
            pre += pred->weights[var*n+idx] * x[i];
            idx++;
        }
        if(xcsf->PRED_TYPE == PRED_TYPE_NLMS_QUADRATIC) {
            // multiply quadratic coefficients with prediction input
            for(int i = 0; i < xcsf->x_dim; i++) {
                for(int j = i; j < xcsf->x_dim; j++) {
                    pre += pred->weights[var*n+idx] * x[i] * x[j];
                    idx++;
                }
            }
        }
        c->prediction[var] = pre;
    }
} 

void pred_nlms_print(const XCSF *xcsf, const CL *c)
{
    const PRED_NLMS *pred = c->pred;
    int n = pred->n;
    printf("eta: %.5f, weights: ", pred->eta);
    for(int var = 0; var < xcsf->y_dim; var++) {
        for(int i = 0; i < n; i++) {
            printf("%f, ", pred->weights[var*n+i]);
        }
        printf("\n");
    }
}

_Bool pred_nlms_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool pred_nlms_mutate(const XCSF *xcsf, const CL *c)
{
    if(xcsf->PRED_EVOLVE_ETA) {
        PRED_NLMS *pred = c->pred;
        sam_adapt(xcsf, pred->mu, N_MU);
        double orig = pred->eta;
        pred->eta += rand_normal(0, pred->mu[0]);
        pred->eta = constrain(ETA_MIN, ETA_MAX, pred->eta);
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
