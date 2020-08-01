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
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include "xcsf.h"
#include "utils.h"
#include "sam.h"
#include "cl.h"
#include "condition.h"
#include "cond_ternary.h"

#define N_MU (1) //!< Number of ternary mutation rates
#define P_DONTCARE (0.5) //!< Don't care probability in randomisation and covering
#define DONT_CARE ('#') //!< Don't care symbol

static void cond_ternary_rand(const XCSF *xcsf, const CL *c);
static void float_to_binary(double f, char *binary, int bits);

void cond_ternary_init(const XCSF *xcsf, CL *c)
{
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    new->length = xcsf->x_dim * xcsf->COND_BITS;
    new->string = malloc(sizeof(char) * new->length);
    new->tmp_input = malloc(sizeof(char) * xcsf->COND_BITS);
    new->mu = malloc(sizeof(double) * N_MU);
    sam_init(xcsf, new->mu, N_MU);
    c->cond = new;
    cond_ternary_rand(xcsf, c);
}

static void cond_ternary_rand(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_TERNARY *cond = c->cond;
    for(int i = 0; i < cond->length; ++i) {
        if(rand_uniform(0, 1) < P_DONTCARE) {
            cond->string[i] = DONT_CARE;
        } else if(rand_uniform(0, 1) < 0.5) {
            cond->string[i] = '0';
        } else {
            cond->string[i] = '1';
        }
    }
}

void cond_ternary_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_TERNARY *cond = c->cond;
    free(cond->string);
    free(cond->tmp_input);
    free(cond->mu);
    free(c->cond);
}

void cond_ternary_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    (void)xcsf;
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    const COND_TERNARY *src_cond = src->cond;
    new->length = src_cond->length;
    new->string = malloc(sizeof(char) * src_cond->length);
    new->tmp_input = malloc(sizeof(char) * xcsf->COND_BITS);
    new->mu = malloc(sizeof(double) * N_MU);
    memcpy(new->string, src_cond->string, sizeof(char) * src_cond->length);
    memcpy(new->mu, src_cond->mu, sizeof(double) * N_MU);
    dest->cond = new;
}

void cond_ternary_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_TERNARY *cond = c->cond;
    for(int i = 0; i < xcsf->x_dim; ++i) {
        float_to_binary(x[i], cond->tmp_input, xcsf->COND_BITS);
        for(int j = 0; j < xcsf->COND_BITS; ++j) {
            if(rand_uniform(0, 1) < P_DONTCARE) {
                cond->string[i * xcsf->COND_BITS + j] = DONT_CARE;
            } else {
                cond->string[i * xcsf->COND_BITS + j] = cond->tmp_input[j];
            }
        }
    }
}

void cond_ternary_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)xcsf;
    (void)c;
    (void)x;
    (void)y;
}

_Bool cond_ternary_match(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_TERNARY *cond = c->cond;
    for(int i = 0; i < xcsf->x_dim; ++i) {
        float_to_binary(x[i], cond->tmp_input, xcsf->COND_BITS);
        for(int j = 0; j < xcsf->COND_BITS; ++j) {
            char s = cond->string[i * xcsf->COND_BITS + j];
            if(s != DONT_CARE && s != cond->tmp_input[j]) {
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
    if(rand_uniform(0, 1) < xcsf->P_CROSSOVER) {
        for(int i = 0; i < cond1->length; ++i) {
            if(rand_uniform(0, 1) < 0.5) {
                char tmp = cond1->string[i];
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
    const COND_TERNARY *cond = c->cond;
    sam_adapt(xcsf, cond->mu, N_MU);
    _Bool changed = false;
    for(int i = 0; i < cond->length; ++i) {
        if(rand_uniform(0, 1) < cond->mu[0]) {
            if(cond->string[i] == DONT_CARE) {
                cond->string[i] = (char) irand_uniform(0, 2) + '0';
            } else {
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
    for(int i = 0; i < cond1->length; ++i) {
        if(cond1->string[i] != DONT_CARE && cond1->string[i] != cond2->string[i]) {
            return false;
        } else if(cond1->string[i] != cond2->string[i]) {
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
    for(int i = 0; i < cond->length; ++i) {
        printf("%c", cond->string[i]);
    }
    printf("\n");
}

int cond_ternary_size(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_TERNARY *cond = c->cond;
    return cond->length;
}

size_t cond_ternary_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    const COND_TERNARY *cond = c->cond;
    s += fwrite(&cond->length, sizeof(int), 1, fp);
    s += fwrite(cond->string, sizeof(char), cond->length, fp);
    s += fwrite(cond->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t cond_ternary_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    COND_TERNARY *new = malloc(sizeof(COND_TERNARY));
    new->length = 0;
    s += fread(&new->length, sizeof(int), 1, fp);
    if(new->length < 1) {
        printf("cond_ternary_load(): read error\n");
        new->length = 1;
        exit(EXIT_FAILURE);
    }
    new->string = malloc(sizeof(char) * new->length);
    s += fread(new->string, sizeof(char), new->length, fp);
    new->tmp_input = malloc(sizeof(char) * xcsf->COND_BITS);
    new->mu = malloc(sizeof(double) * N_MU);
    s += fread(new->mu, sizeof(double), N_MU, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Generates a binary string from a float.
 * @param f The float to binarise.
 * @param binary The converted binary string (set by this function).
 * @param bits The number of bits to use for binarising.
 */
static void float_to_binary(double f, char *binary, int bits)
{
    int a = (int)(f * pow(2, bits));
    for(int i = 0; i < bits; ++i) {
        binary[i] = (a % 2) + '0';
        a /= 2;
    }
}
