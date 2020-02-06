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
 * @file act_integer.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief integer action functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"       
#include "utils.h"
#include "sam.h"
#include "action.h"
#include "act_integer.h"
 
#define N_MU 1 //!< Number of integer action mutation rates

/**
 * @brief Integer action data structure.
 */
typedef struct ACT_INTEGER {
    int action; //!< Integer action
    double mu[N_MU]; //!< Mutation rates
} ACT_INTEGER;

_Bool act_integer_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool act_integer_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    const ACT_INTEGER *act1 = c1->act;
    const ACT_INTEGER *act2 = c2->act;
    if(act1->action != act2->action) {
        return false;
    }
    return true;
}

_Bool act_integer_mutate(const XCSF *xcsf, const CL *c)
{
    ACT_INTEGER *act = c->act;
    if(rand_uniform(0,1) < act->mu[0]) {
        int old = act->action;
        act->action = irand_uniform(0, xcsf->n_actions);
        if(old != act->action) {
            return true;
        }
    }
    return false;
}

int act_integer_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    (void)xcsf; (void)x;
    const ACT_INTEGER *act = c->act;
    return act->action;
}

void act_integer_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    (void)xcsf;
    ACT_INTEGER *new = malloc(sizeof(ACT_INTEGER));
    const ACT_INTEGER *from_act = from->act;
    new->action = from_act->action;
    memcpy(new->mu, from_act->mu, N_MU * sizeof(double));
    to->act = new;
}

void act_integer_print(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const ACT_INTEGER *act = c->act;
    printf("%d\n", act->action);
}

void act_integer_rand(const XCSF *xcsf, const CL *c)
{
    ACT_INTEGER *act = c->act;
    act->action = irand_uniform(0, xcsf->n_actions);
}
 
void act_integer_cover(const XCSF *xcsf, const CL *c, const double *x, int action)
{
    (void)xcsf; (void)x;
    ACT_INTEGER *act = c->act;
    act->action = action;
}

void act_integer_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    free(c->act);
}

void act_integer_init(const XCSF *xcsf, CL *c)
{
    ACT_INTEGER *new = malloc(sizeof(ACT_INTEGER));
    sam_init(xcsf, new->mu, N_MU);
    c->act = new;
    act_integer_rand(xcsf, c);
}
 
void act_integer_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

size_t act_integer_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    ACT_INTEGER *act = c->act;
    s += fwrite(&act->action, sizeof(int), 1, fp);
    s += fwrite(act->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t act_integer_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    ACT_INTEGER *new = malloc(sizeof(ACT_INTEGER));
    s += fread(&new->action, sizeof(int), 1, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->act = new;
    return s;
}
