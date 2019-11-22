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
   
/**
 * @file cond_dgp.c
 * @brief Dynamical GP graph condition functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "dgp.h"
#include "cl.h"
#include "condition.h"
#include "cond_dgp.h"

/**
 * @brief Dynamical GP graph condition data structure.
 */ 
typedef struct COND_DGP {
    GRAPH dgp; //!< DGP graph
} COND_DGP;

void cond_dgp_rand(XCSF *xcsf, CL *c);

void cond_dgp_init(XCSF *xcsf, CL *c)
{
    COND_DGP *new = malloc(sizeof(COND_DGP));
    graph_init(xcsf, &new->dgp, xcsf->DGP_NUM_NODES);
    graph_rand(xcsf, &new->dgp);
    c->cond = new;
}

void cond_dgp_free(XCSF *xcsf, CL *c)
{
    COND_DGP *cond = c->cond;
    graph_free(xcsf, &cond->dgp);
    free(c->cond);
}                  

void cond_dgp_copy(XCSF *xcsf, CL *to, CL *from)
{
    COND_DGP *new = malloc(sizeof(COND_DGP));
    COND_DGP *from_cond = from->cond;
    graph_init(xcsf, &new->dgp, from_cond->dgp.n);
    graph_copy(xcsf, &new->dgp, &from_cond->dgp);
    to->cond = new;
}

void cond_dgp_rand(XCSF *xcsf, CL *c)
{
    COND_DGP *cond = c->cond;
    graph_rand(xcsf, &cond->dgp);
}

void cond_dgp_cover(XCSF *xcsf, CL *c, double *x)
{
    do {
        cond_dgp_rand(xcsf, c);
    } while(!cond_dgp_match(xcsf, c, x));
}
 
void cond_dgp_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}
 
_Bool cond_dgp_match(XCSF *xcsf, CL *c, double *x)
{
    COND_DGP *cond = c->cond;
    graph_update(xcsf, &cond->dgp, x);
    if(graph_output(xcsf, &cond->dgp, 0) > 0.5) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}            

_Bool cond_dgp_mutate(XCSF *xcsf, CL *c)
{
    COND_DGP *cond = c->cond;
    return graph_mutate(xcsf, &cond->dgp);
}

_Bool cond_dgp_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    COND_DGP *cond1 = c1->cond;
    COND_DGP *cond2 = c2->cond;
    return graph_crossover(xcsf, &cond1->dgp, &cond2->dgp);
}

_Bool cond_dgp_general(XCSF *xcsf, CL *c1, CL *c2)
{
    if(c1->exp < xcsf->THETA_SUB || c2->exp < xcsf->THETA_SUB) {
        return false;
    }
    for(int i = 0; i < xcsf->THETA_SUB; i++) {
        int i1 = (c1->exp + i) % xcsf->THETA_SUB;
        int i2 = (c2->exp + i) % xcsf->THETA_SUB;
        if(c1->mhist[i1] == false && c2->mhist[i2] == true) {
            return false;
        }
    }
    return true;
}

void cond_dgp_print(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_DGP *cond = c->cond;
    graph_print(xcsf, &cond->dgp);
}  

int cond_dgp_size(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_DGP *cond = c->cond;
    return cond->dgp.n;
}

size_t cond_dgp_save(XCSF *xcsf, CL *c, FILE *fp)
{
    COND_DGP *cond = c->cond;
    size_t s = graph_save(xcsf, &cond->dgp, fp);
    //printf("cond dgp saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t cond_dgp_load(XCSF *xcsf, CL *c, FILE *fp)
{
    COND_DGP *new = malloc(sizeof(COND_DGP));
    size_t s = graph_load(xcsf, &new->dgp, fp);
    c->cond = new;
    //printf("cond dgp loaded %lu elements\n", (unsigned long)s);
    return s;
}
