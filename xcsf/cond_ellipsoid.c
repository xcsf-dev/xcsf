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
 * @file cond_ellipsoid.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Hyperellipsoid condition functions.
 */

#include "cond_ellipsoid.h"
#include "sam.h"
#include "utils.h"

#define N_MU (1) //!< Number of hyperellipsoid mutation rates
static const int MU_TYPE[N_MU] = { SAM_LOG_NORMAL }; //<! Self-adaptation method

static double
cond_ellipsoid_dist(const struct XCSF *xcsf, const struct CL *c,
                    const double *x)
{
    const struct COND_ELLIPSOID *cond = c->cond;
    double dist = 0;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        double d = (x[i] - cond->center[i]) / cond->spread[i];
        dist += d * d;
    }
    return dist;
}

/**
 * @brief Creates and initialises a hyperellipsoid condition.
 * @details Uses the center-spread representation.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose condition is to be initialised.
 */
void
cond_ellipsoid_init(const struct XCSF *xcsf, struct CL *c)
{
    struct COND_ELLIPSOID *new = malloc(sizeof(struct COND_ELLIPSOID));
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    double spread_max = fabs(xcsf->COND_MAX - xcsf->COND_MIN);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        new->center[i] = rand_uniform(xcsf->COND_MIN, xcsf->COND_MAX);
        new->spread[i] = rand_uniform(xcsf->COND_SMIN, spread_max);
    }
    sam_init(new->mu, N_MU, MU_TYPE);
    c->cond = new;
}

void
cond_ellipsoid_free(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    const struct COND_ELLIPSOID *cond = c->cond;
    free(cond->center);
    free(cond->spread);
    free(cond->mu);
    free(c->cond);
}

void
cond_ellipsoid_copy(const struct XCSF *xcsf, struct CL *dest,
                    const struct CL *src)
{
    struct COND_ELLIPSOID *new = malloc(sizeof(struct COND_ELLIPSOID));
    const struct COND_ELLIPSOID *src_cond = src->cond;
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    memcpy(new->center, src_cond->center, sizeof(double) * xcsf->x_dim);
    memcpy(new->spread, src_cond->spread, sizeof(double) * xcsf->x_dim);
    memcpy(new->mu, src_cond->mu, sizeof(double) * N_MU);
    dest->cond = new;
}

void
cond_ellipsoid_cover(const struct XCSF *xcsf, const struct CL *c,
                     const double *x)
{
    const struct COND_ELLIPSOID *cond = c->cond;
    double spread_max = fabs(xcsf->COND_MAX - xcsf->COND_MIN);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        cond->center[i] = x[i];
        cond->spread[i] = rand_uniform(xcsf->COND_SMIN, spread_max);
    }
}

void
cond_ellipsoid_update(const struct XCSF *xcsf, const struct CL *c,
                      const double *x, const double *y)
{
    (void) y;
    if (xcsf->COND_ETA > 0) {
        const struct COND_ELLIPSOID *cond = c->cond;
        for (int i = 0; i < xcsf->x_dim; ++i) {
            cond->center[i] += xcsf->COND_ETA * (x[i] - cond->center[i]);
        }
    }
}

_Bool
cond_ellipsoid_match(const struct XCSF *xcsf, const struct CL *c,
                     const double *x)
{
    if (cond_ellipsoid_dist(xcsf, c, x) < 1) {
        return true;
    }
    return false;
}

_Bool
cond_ellipsoid_crossover(const struct XCSF *xcsf, const struct CL *c1,
                         const struct CL *c2)
{
    const struct COND_ELLIPSOID *cond1 = c1->cond;
    const struct COND_ELLIPSOID *cond2 = c2->cond;
    _Bool changed = false;
    if (rand_uniform(0, 1) < xcsf->P_CROSSOVER) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            if (rand_uniform(0, 1) < 0.5) {
                double tmp = cond1->center[i];
                cond1->center[i] = cond2->center[i];
                cond2->center[i] = tmp;
                changed = true;
            }
            if (rand_uniform(0, 1) < 0.5) {
                double tmp = cond1->spread[i];
                cond1->spread[i] = cond2->spread[i];
                cond2->spread[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

_Bool
cond_ellipsoid_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    _Bool changed = false;
    const struct COND_ELLIPSOID *cond = c->cond;
    double *center = cond->center;
    double *spread = cond->spread;
    sam_adapt(cond->mu, N_MU, MU_TYPE);
    for (int i = 0; i < xcsf->x_dim; ++i) {
        double orig = center[i];
        center[i] += rand_normal(0, cond->mu[0]);
        center[i] = clamp(center[i], xcsf->COND_MIN, xcsf->COND_MAX);
        if (orig != center[i]) {
            changed = true;
        }
        orig = spread[i];
        spread[i] += rand_normal(0, cond->mu[0]);
        if (orig != spread[i]) {
            changed = true;
        }
    }
    return changed;
}

_Bool
cond_ellipsoid_general(const struct XCSF *xcsf, const struct CL *c1,
                       const struct CL *c2)
{
    const struct COND_ELLIPSOID *cond1 = c1->cond;
    const struct COND_ELLIPSOID *cond2 = c2->cond;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        double l1 = cond1->center[i] - cond1->spread[i];
        double l2 = cond2->center[i] - cond2->spread[i];
        double u1 = cond1->center[i] + cond1->spread[i];
        double u2 = cond2->center[i] + cond2->spread[i];
        if (l1 > l2 || u1 < u2) {
            return false;
        }
    }
    return true;
}

void
cond_ellipsoid_print(const struct XCSF *xcsf, const struct CL *c)
{
    const struct COND_ELLIPSOID *cond = c->cond;
    printf("ellipsoid:");
    for (int i = 0; i < xcsf->x_dim; ++i) {
        printf(" (%5f, ", cond->center[i]);
        printf("%5f)", cond->spread[i]);
    }
    printf("\n");
}

double
cond_ellipsoid_size(const struct XCSF *xcsf, const struct CL *c)
{
    (void) c;
    return xcsf->x_dim;
}

size_t
cond_ellipsoid_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    size_t s = 0;
    const struct COND_ELLIPSOID *cond = c->cond;
    s += fwrite(cond->center, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->spread, sizeof(double), xcsf->x_dim, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t
cond_ellipsoid_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    size_t s = 0;
    struct COND_ELLIPSOID *new = malloc(sizeof(struct COND_ELLIPSOID));
    new->center = malloc(sizeof(double) * xcsf->x_dim);
    new->spread = malloc(sizeof(double) * xcsf->x_dim);
    new->mu = malloc(sizeof(double) * N_MU);
    s += fread(new->center, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->spread, sizeof(double), xcsf->x_dim, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}
