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
   
/**
 * @file act_integer.c
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
 
_Bool act_integer_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool act_integer_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf;
    if(c1->action != c2->action) {
        return false;
    }
    return true;
}

_Bool act_integer_mutate(XCSF *xcsf, CL *c)
{
    if(rand_uniform(0,1) < xcsf->P_MUTATION) {
        int old = c->action;
        c->action = irand_uniform(0, xcsf->num_actions);
        if(old != c->action) {
            return true;
        }
    }
    return false;
}

int act_integer_compute(XCSF *xcsf, CL *c, double *x)
{
    (void)xcsf; (void)x;
    return c->action;
}

void act_integer_copy(XCSF *xcsf, CL *to, CL *from)
{
    (void)xcsf;
    to->action = from->action;
}

void act_integer_print(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    printf("%d\n", c->action);
}

void act_integer_rand(XCSF *xcsf, CL *c)
{
    c->action = irand_uniform(0, xcsf->num_actions);
}
 
void act_integer_cover(XCSF *xcsf, CL *c, int action)
{
    (void)xcsf;
    c->action = action;
}

void act_integer_free(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void act_integer_init(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}
 
void act_integer_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

size_t act_integer_save(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&c->action, sizeof(int), 1, fp);
    //printf("integer saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t act_integer_load(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&c->action, sizeof(int), 1, fp);
    //printf("integer loaded %lu elements\n", (unsigned long)s);
    return s;
}
