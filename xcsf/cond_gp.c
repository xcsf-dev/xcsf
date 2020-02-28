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
 * @file cond_gp.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Tree GP condition functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "xcsf.h"
#include "utils.h"
#include "sam.h"
#include "cl.h"
#include "condition.h"
#include "cond_gp.h"
#include "gp.h"

/**
 * @brief Tree GP condition data structure.
 */ 
typedef struct COND_GP {
    GP_TREE gp; //!< GP tree
} COND_GP;

static void cond_gp_rand(const XCSF *xcsf, const CL *c);

void cond_gp_init(const XCSF *xcsf, CL *c)
{
    COND_GP *new = malloc(sizeof(COND_GP));
    tree_rand(xcsf, &new->gp);
    c->cond = new;
}

void cond_gp_free(const XCSF *xcsf, const CL *c)
{
    const COND_GP *cond = c->cond;
    tree_free(xcsf, &cond->gp);
    free(c->cond);
}

void cond_gp_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    COND_GP *new = malloc(sizeof(COND_GP));
    const COND_GP *src_cond = src->cond;
    tree_copy(xcsf, &new->gp, &src_cond->gp);
    dest->cond = new;
}

static void cond_gp_rand(const XCSF *xcsf, const CL *c)
{
    COND_GP *cond = c->cond;
    tree_free(xcsf, &cond->gp);
    tree_rand(xcsf, &cond->gp);
}
 
void cond_gp_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    do {
        cond_gp_rand(xcsf, c);
    } while(!cond_gp_match(xcsf, c, x));
}
 
void cond_gp_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool cond_gp_match(const XCSF *xcsf, const CL *c, const double *x)
{
    COND_GP *cond = c->cond;
    cond->gp.p = 0;
    double result = tree_eval(xcsf, &cond->gp, x);
    if(result > 0.5) {
        return true;
    }
    return false;
}    

_Bool cond_gp_mutate(const XCSF *xcsf, const CL *c)
{
    COND_GP *cond = c->cond;
    return tree_mutate(xcsf, &cond->gp);
}

_Bool cond_gp_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    COND_GP *cond1 = c1->cond;
    COND_GP *cond2 = c2->cond;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        tree_crossover(xcsf, &cond1->gp, &cond2->gp);
        return true;
    }
    return false;
}

_Bool cond_gp_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

void cond_gp_print(const XCSF *xcsf, const CL *c)
{
    const COND_GP *cond = c->cond;
    printf("GP tree: ");
    tree_print(xcsf, &cond->gp, 0);
    printf("\n");
}  

int cond_gp_size(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_GP *cond = c->cond;
    return cond->gp.len;
}

size_t cond_gp_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    const COND_GP *cond = c->cond;
    size_t s = tree_save(xcsf, &cond->gp, fp);
    return s;
}

size_t cond_gp_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    COND_GP *new = malloc(sizeof(COND_GP));
    size_t s = tree_load(xcsf, &new->gp, fp);
    c->cond = new;
    return s;
}
