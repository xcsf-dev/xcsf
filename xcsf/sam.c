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
 **************
 * Description: 
 **************
 * The self-adaptive mutation module.
 *
 * Initialises the classifier mutation rates and performs self-adaptation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "sam.h"

void sam_log_normal_init(XCSF *xcsf, double **mu);
void sam_log_normal_adapt(XCSF *xcsf, double *mu);

#define NUM_RATES 10
static const double mrates[NUM_RATES] = {0.0001,0.001,0.002,0.005,0.01,0.01,0.02,0.05,0.1,1.0}; 
void sam_rate_selection_init(XCSF *xcsf, double **mu);
void sam_rate_selection_adapt(XCSF *xcsf, double *mu);

void sam_init(XCSF *xcsf, double **mu)
{
    if(xcsf->SAM_NUM > 0) {
        *mu = malloc(sizeof(double) * xcsf->SAM_NUM);
        if(xcsf->SAM_TYPE == 0) {
            sam_log_normal_init(xcsf, mu);
        }
        else {
            sam_rate_selection_init(xcsf, mu);
        }
    }
}

void sam_adapt(XCSF *xcsf, double *mu)
{
    if(xcsf->SAM_TYPE == 0) {
        sam_log_normal_adapt(xcsf, mu);
    }
    else {
        sam_rate_selection_adapt(xcsf, mu);
    }
}

void sam_copy(XCSF *xcsf, double *to, double *from)
{
    if(xcsf->SAM_NUM > 0) {
        memcpy(to, from, sizeof(double) * xcsf->SAM_NUM);
    }
}

void sam_free(XCSF *xcsf, double *mu)
{
    if(xcsf->SAM_NUM > 0) {
        free(mu);
    }
}

void sam_print(XCSF *xcsf, double *mu)
{
    printf("mu: \n");
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        printf("%f, ", mu[i]);
    }
    printf("\n");
}

void sam_log_normal_init(XCSF *xcsf, double **mu)
{
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        (*mu)[i] = rand_uniform(0,1);
    }
}

void sam_log_normal_adapt(XCSF *xcsf, double *mu)
{
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        mu[i] *= exp(rand_normal(0,1));
        if(mu[i] < xcsf->SAM_MIN) {
            mu[i] = xcsf->SAM_MIN;
        }
        else if(mu[i] > 1.0) {
            mu[i] = 1.0;
        }
    }
}
 
void sam_rate_selection_init(XCSF *xcsf, double **mu)
{
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        (*mu)[i] = mrates[irand_uniform(0,NUM_RATES)];
    }
}
 
void sam_rate_selection_adapt(XCSF *xcsf, double *mu)
{
    for(int i = 0; i < xcsf->SAM_NUM; i++) {
        if(rand_uniform(0,1) < 0.1) {
            mu[i] = mrates[irand_uniform(0,NUM_RATES)];
        }
    }
} 
