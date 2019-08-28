/*
 * Copyright (C) 2016--2019 Richard Preen <rpreen@gmail.com>
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
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "condition.h"
#include "cond_gp.h"
#include "gp.h"

typedef struct COND_GP {
    GP_TREE gp;
} COND_GP;

void cond_gp_rand(XCSF *xcsf, CL *c);

void cond_gp_init(XCSF *xcsf, CL *c)
{
    COND_GP *new = malloc(sizeof(COND_GP));
    tree_rand(xcsf, &new->gp);
    c->cond = new;
}

void cond_gp_free(XCSF *xcsf, CL *c)
{
    COND_GP *cond = c->cond;
    tree_free(xcsf, &cond->gp);
    free(c->cond);
}

void cond_gp_copy(XCSF *xcsf, CL *to, CL *from)
{
    COND_GP *new = malloc(sizeof(COND_GP));
    COND_GP *from_cond = from->cond;
    tree_copy(xcsf, &new->gp, &from_cond->gp);
    to->cond = new;
}

void cond_gp_rand(XCSF *xcsf, CL *c)
{
    COND_GP *cond = c->cond;
    tree_free(xcsf, &cond->gp);
    tree_rand(xcsf, &cond->gp);
}
 
void cond_gp_cover(XCSF *xcsf, CL *c, double *x)
{
    do {
        cond_gp_rand(xcsf, c);
    } while(!cond_gp_match(xcsf, c, x));
}
 
void cond_gp_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool cond_gp_match(XCSF *xcsf, CL *c, double *x)
{
    COND_GP *cond = c->cond;
    cond->gp.p = 0;
    double result = tree_eval(xcsf, &cond->gp, x);
    if(result > 0.5) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}    

_Bool cond_gp_mutate(XCSF *xcsf, CL *c)
{
    COND_GP *cond = c->cond;
    if(rand_uniform(0,1) < xcsf->P_MUTATION) {
        tree_mutation(xcsf, &cond->gp, xcsf->P_MUTATION);
        return true;
    }
    else {
        return false;
    }
}

_Bool cond_gp_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    COND_GP *cond1 = c1->cond;
    COND_GP *cond2 = c2->cond;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        tree_crossover(xcsf, &cond1->gp, &cond2->gp);
        return true;
    }
    else {
        return false;
    }
}

_Bool cond_gp_subsumes(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool cond_gp_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}   

void cond_gp_print(XCSF *xcsf, CL *c)
{
    COND_GP *cond = c->cond;
    printf("GP tree: ");
    tree_print(xcsf, &cond->gp, 0);
    printf("\n");
}  
