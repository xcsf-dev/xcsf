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
 * @date 2019--2020.
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
#include "sam.h"
#include "cl.h"
#include "condition.h"
#include "cond_ternary.h"

#define N_MU 1 //!< Number of ternary mutation rates
#define P_DONTCARE 0.5 //!< Don't care probability in randomisation and covering
#define DONT_CARE '#' //!< Don't care symbol

/**
 * @brief Ternary condition data structure.
 */ 
typedef struct COND_TERNARY {
    char *string; //!< Ternary bitstring
    int len; //!< Length of the bitstring
    double mu[N_MU]; //!< Mutation rates
} COND_TERNARY;

static void cond_ternary_rand(const XCSF *xcsf, const CL *c);

void cond_ternary_init(const XCSF *xcsf, CL *c)
{
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    new->len = xcsf->x_dim * xcsf->COND_BITS;
    new->string = malloc(sizeof(char) * new->len);
    sam_init(xcsf, new->mu, N_MU);
    c->cond = new;     
    cond_ternary_rand(xcsf, c);
}

static void cond_ternary_rand(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_TERNARY *cond = c->cond;
    for(int i = 0; i < cond->len; i++) {
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

void cond_ternary_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_TERNARY *cond = c->cond;
    free(cond->string);
    free(c->cond);
}

void cond_ternary_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    (void)xcsf;
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    const COND_TERNARY *from_cond = from->cond;
    new->len = from_cond->len;
    new->string = malloc(sizeof(char) * from_cond->len);
    memcpy(new->string, from_cond->string, sizeof(char) * from_cond->len);
    memcpy(new->mu, from_cond->mu, sizeof(double) * N_MU);
    to->cond = new;
}                             

void cond_ternary_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_TERNARY *cond = c->cond;
    char state[xcsf->COND_BITS];
    for(int i = 0; i < xcsf->x_dim; i++) {
        float_to_binary(x[i], state, xcsf->COND_BITS);
        for(int b = 0; b < xcsf->COND_BITS; b++) {
            if(rand_uniform(0,1) < P_DONTCARE) {
                cond->string[i*xcsf->COND_BITS+b] = DONT_CARE;
            }
            else {
                cond->string[i*xcsf->COND_BITS+b] = state[b];
            }
        }
    }
}

void cond_ternary_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool cond_ternary_match(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_TERNARY *cond = c->cond;
    char state[xcsf->COND_BITS];
    for(int i = 0; i < xcsf->x_dim; i++) {
        float_to_binary(x[i], state, xcsf->COND_BITS);
        for(int b = 0; b < xcsf->COND_BITS; b++) {
            char s = cond->string[i*xcsf->COND_BITS+b];
            if(s != DONT_CARE && s != state[b]) {
                return false;
            }
        }
    }
    return true;
}

_Bool cond_ternary_crossover(const XCSF *xcsf, const CL *c1, const CL *c2) 
{
    const COND_TERNARY *cond1 = c1->cond;
    const COND_TERNARY *cond2 = c2->cond;
    _Bool changed = false;
    if(rand_uniform(0,1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < cond1->len; i++) {
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

_Bool cond_ternary_mutate(const XCSF *xcsf, const CL *c)
{
    COND_TERNARY *cond = c->cond;
    sam_adapt(xcsf, cond->mu, N_MU);
    _Bool changed = false;
    for(int i = 0; i < cond->len; i++) {
        if(rand_uniform(0,1) < cond->mu[0]) {
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

_Bool cond_ternary_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    const COND_TERNARY *cond1 = c1->cond;
    const COND_TERNARY *cond2 = c2->cond;
    _Bool general = false;
    for(int i = 0; i < cond1->len; i++) {
        if(cond1->string[i] != DONT_CARE && cond1->string[i] != cond2->string[i]) {
            return false;
        }
        else if(cond1->string[i] != cond2->string[i]) {
            general = true;
        }
    }
    return general;
}  

void cond_ternary_print(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_TERNARY *cond = c->cond;
    printf("ternary:");
    for(int i = 0; i < cond->len; i++) {
        printf("%c", cond->string[i]);
    }
    printf("\n");
}

int cond_ternary_size(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_TERNARY *cond = c->cond;
    return cond->len;
}

size_t cond_ternary_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    const COND_TERNARY *cond = c->cond;
    s += fwrite(&cond->len, sizeof(int), 1, fp);
    s += fwrite(cond->string, sizeof(char), cond->len, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t cond_ternary_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    s += fread(&new->len, sizeof(int), 1, fp);
    new->string = malloc(sizeof(char) * new->len);
    s += fread(new->string, sizeof(char), new->len, fp);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}
