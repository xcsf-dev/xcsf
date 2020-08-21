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
 * @file pred_constant.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Piece-wise constant prediction functions.
 */

#include "pred_constant.h"

void
pred_constant_init(const struct XCSF *xcsf, struct CL *c)
{
    (void) xcsf;
    (void) c;
}

void
pred_constant_copy(const struct XCSF *xcsf, struct CL *dest,
                   const struct CL *src)
{
    (void) xcsf;
    (void) dest;
    (void) src;
}

void
pred_constant_free(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
}

void
pred_constant_update(const struct XCSF *xcsf, const struct CL *c,
                     const double *x, const double *y)
{
    (void) x;
    if (c->exp * xcsf->PRED_ETA < 1) {
        for (int var = 0; var < xcsf->y_dim; ++var) {
            c->prediction[var] =
                (c->prediction[var] * (c->exp - 1) + y[var]) / c->exp;
        }
    } else {
        for (int var = 0; var < xcsf->y_dim; ++var) {
            c->prediction[var] +=
                xcsf->PRED_ETA * (y[var] - c->prediction[var]);
        }
    }
}

void
pred_constant_compute(const struct XCSF *xcsf, const struct CL *c,
                      const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
}

void
pred_constant_print(const struct XCSF *xcsf, const struct CL *c)
{
    printf("constant prediction: %f", c->prediction[0]);
    for (int var = 1; var < xcsf->y_dim; ++var) {
        printf(", %f", c->prediction[var]);
    }
    printf("\n");
}

_Bool
pred_constant_crossover(const struct XCSF *xcsf, const struct CL *c1,
                        const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
pred_constant_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

double
pred_constant_size(const struct XCSF *xcsf, const struct CL *c)
{
    (void) c;
    return xcsf->y_dim;
}

size_t
pred_constant_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

size_t
pred_constant_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}
