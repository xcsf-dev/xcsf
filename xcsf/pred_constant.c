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
 * @date 2015--2019.
 * @brief Piece-wise constant prediction functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "prediction.h"
#include "pred_constant.h"

void pred_constant_init(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void pred_constant_copy(XCSF *xcsf, CL *to, CL *from)
{
    (void)xcsf; (void)to; (void)from;
}

void pred_constant_free(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void pred_constant_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)x;
    if(c->exp < 1.0 / xcsf->PRED_ETA) {
        for(int var = 0; var < xcsf->num_y_vars; var++) {
            c->prediction[var] = (c->prediction[var] * (c->exp-1.0) + y[var]) / (double)c->exp;
        }
    }
    else {
        for(int var = 0; var < xcsf->num_y_vars; var++) {
            c->prediction[var] += xcsf->PRED_ETA * (y[var] - c->prediction[var]);
        }
    }
}

double *pred_constant_compute(XCSF *xcsf, CL *c, double *x)
{
    (void)xcsf; (void)c; (void)x;
    return c->prediction;
} 

void pred_constant_print(XCSF *xcsf, CL *c)
{
    printf("predictions: ");
    printf(", %f", c->prediction[0]);
    for(int var = 1; var < xcsf->num_y_vars; var++) {
        printf(", %f", c->prediction[var]);
    }
    printf("\n");
}

_Bool pred_constant_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool pred_constant_mutate(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
    return false;
}

int pred_constant_size(XCSF *xcsf, CL *c)
{
    (void)c;
    return xcsf->num_y_vars;
}

size_t pred_constant_save(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}

size_t pred_constant_load(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}
