/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
 *
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
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"       
#include "utils.h"
#include "action.h"
#include "act_constant.h"
 
_Bool act_constant_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool act_constant_general(XCSF *xcsf, CL *c1, CL *c2)
{
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        if(c1->action[i] != c2->action[i]) {
            return false;
        }
    }
    return true;
}

_Bool act_constant_mutate(XCSF *xcsf, CL *c)
{
    if(xcsf->num_classes > 0) {
        if(rand_uniform(0,1) < xcsf->P_MUTATION) {
            act_constant_rand(xcsf, c);
            return true;
        }
    }
    return false;
}

double *act_constant_compute(XCSF *xcsf, CL *c, double *x)
{
    (void)xcsf; (void)x;
    return c->action;
}

void act_constant_copy(XCSF *xcsf, CL *to, CL *from)
{
    memcpy(to->action, from->action, sizeof(double)*xcsf->num_y_vars);
}

void act_constant_free(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void act_constant_init(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void act_constant_print(XCSF *xcsf, CL *c)
{
    printf("%.1f", c->action[0]);
    for(int i = 1; i < xcsf->num_y_vars; i++) {
        printf(", %.1f", c->action[i]);
    }
    printf("\n");
}

void act_constant_rand(XCSF *xcsf, CL *c)
{
    int class = irand_uniform(0, xcsf->num_classes);
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        if(i == class) {
            c->action[i] = 1;
        }
        else {
            c->action[i] = 0;
        }
    }          
}

void act_constant_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}
