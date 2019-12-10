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
 * @file cond_ternary.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019.
 * @brief Ternary condition functions.
 * @details Binarises inputs.
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
#include "cond_ternary.h"

#define P_DONTCARE 0.5
#define DONT_CARE '#'

/**
 * @brief Ternary condition data structure.
 */ 
typedef struct COND_TERNARY {
    char *string; //!< Ternary bitstring
} COND_TERNARY;

static void cond_ternary_rand(XCSF *xcsf, CL *c);

void cond_ternary_init(XCSF *xcsf, CL *c)
{
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    new->string = malloc(sizeof(char) * xcsf->num_x_vars);
    c->cond = new;     
    cond_ternary_rand(xcsf, c);
}

static void cond_ternary_rand(XCSF *xcsf, CL *c)
{
    COND_TERNARY *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        if(rand_uniform(0,1) < P_DONTCARE) {
            cond->string[i] = DONT_CARE;
        }
        else {
            if(rand_uniform(0,1) < 0.5) {
                cond->string[i] = '0';
            }
            else {
                cond->string[i] = '1';
            }
        }
    }
}

void cond_ternary_free(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    COND_TERNARY *cond = c->cond;
    free(cond->string);
    free(c->cond);
}

void cond_ternary_copy(XCSF *xcsf, CL *to, CL *from)
{
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    COND_TERNARY *from_cond = from->cond;
    new->string = malloc(sizeof(char) * xcsf->num_x_vars);
    memcpy(new->string, from_cond->string, sizeof(char) * xcsf->num_x_vars);
    to->cond = new;
}                             

void cond_ternary_cover(XCSF *xcsf, CL *c, double *x)
{
    COND_TERNARY *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        if(rand_uniform(0,1) < P_DONTCARE) {
            cond->string[i] = DONT_CARE;
        }
        else {
            if(x[i] > 0.5) {
                cond->string[i] = '1';
            }
            else {
                cond->string[i] = '0';
            }
        }
    }
}

void cond_ternary_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool cond_ternary_match(XCSF *xcsf, CL *c, double *x)
{
    COND_TERNARY *cond = c->cond;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        char state = '0';
        if(x[i] > 0.5) {
            state = '1';
        }
        if(cond->string[i] != DONT_CARE && cond->string[i] != state) {
            c->m = false;
            return false;
        }
    }
    c->m = true;
    return true;
}

_Bool cond_ternary_crossover(XCSF *xcsf, CL *c1, CL *c2) 
{
    COND_TERNARY *cond1 = c1->cond;
    COND_TERNARY *cond2 = c2->cond;
    _Bool changed = false;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < xcsf->num_x_vars; i++) {
            if(rand_uniform(0,1) < 0.5) {
                double tmp = cond1->string[i];
                cond1->string[i] = cond2->string[i];
                cond2->string[i] = tmp;
                changed = true;
            }
        }
    }
    return changed;
}

_Bool cond_ternary_mutate(XCSF *xcsf, CL *c)
{
    COND_TERNARY *cond = c->cond;
    _Bool changed = false;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        if(rand_uniform(0,1) < xcsf->P_MUTATION) {
            if(cond->string[i] == DONT_CARE) {
                if(rand_uniform(0,1) < 0.5) {
                    cond->string[i] = '1';
                }
                else {
                    cond->string[i] = '0';
                }
            }
            else {
                cond->string[i] = DONT_CARE;
            }
            changed = true;
        }
    }
    return changed;
}

_Bool cond_ternary_general(XCSF *xcsf, CL *c1, CL *c2)
{
    // returns whether cond1 is more general than cond2
    COND_TERNARY *cond1 = c1->cond;
    COND_TERNARY *cond2 = c2->cond;
    _Bool general = false;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        if(cond1->string[i] != DONT_CARE && cond1->string[i] != cond2->string[i]) {
            return false;
        }
        else if(cond1->string[i] != cond2->string[i]) {
            general = true;
        }
    }
    return general;
}  

void cond_ternary_print(XCSF *xcsf, CL *c)
{
    COND_TERNARY *cond = c->cond;
    printf("ternary:");
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        printf("%c", cond->string[i]);
    }
    printf("\n");
}

int cond_ternary_size(XCSF *xcsf, CL *c)
{
    (void)c;
    return xcsf->num_x_vars;
}

size_t cond_ternary_save(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    COND_TERNARY *cond = c->cond;
    s += fwrite(cond->string, sizeof(char), xcsf->num_x_vars, fp);
    return s;
}

size_t cond_ternary_load(XCSF *xcsf, CL *c, FILE *fp)
{
    size_t s = 0;
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    new->string = malloc(sizeof(char) * xcsf->num_x_vars);
    s += fread(new->string, sizeof(char), xcsf->num_x_vars, fp);
    c->cond = new;
    return s;
}
