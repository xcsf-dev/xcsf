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
#include "action.h"
#include "act_integer.h"
 
_Bool act_integer_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool act_integer_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    const int *act1 = c1->act;
    const int *act2 = c2->act;
    if(*act1 != *act2) {
        return false;
    }
    return true;
}

_Bool act_integer_mutate(const XCSF *xcsf, CL *c)
{
    if(rand_uniform(0,1) < xcsf->P_MUTATION) {
        int *act = c->act;
        int old = *act;
        *act = irand_uniform(0, xcsf->num_actions);
        if(old != *act) {
            return true;
        }
    }
    return false;
}

int act_integer_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    (void)xcsf; (void)x;
    const int *act = c->act;
    return *act;
}

void act_integer_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    (void)xcsf;
    int *new = malloc(sizeof(int));
    const int *from_act = from->act;
    *new = *from_act;
    to->act = new;
}

void act_integer_print(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const int *act = c->act;
    printf("%d\n", *act);
}

void act_integer_rand(const XCSF *xcsf, const CL *c)
{
    int *act = c->act;
    *act = irand_uniform(0, xcsf->num_actions);
}
 
void act_integer_cover(const XCSF *xcsf, const CL *c, const double *x, int action)
{
    (void)xcsf; (void)x;
    int *act = c->act;
    *act = action;
}

void act_integer_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf; (void)c;
}

void act_integer_init(const XCSF *xcsf, CL *c)
{
    int *new = malloc(sizeof(int));
    c->act = new;
    act_integer_rand(xcsf, c);
}
 
void act_integer_update(const XCSF *xcsf, CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

size_t act_integer_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf;
    const int *act = c->act;
    size_t s = fwrite(act, sizeof(int), 1, fp);
    return s;
}

size_t act_integer_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    int *act = c->act;
    size_t s = fread(act, sizeof(int), 1, fp);
    return s;
}
