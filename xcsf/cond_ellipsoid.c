/*
 * Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
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
 * The hyperellipsoid classifier condition module.
 *
 * Provides functionality to create real-valued hyperellipsoid conditions
 * whereby a classifier matches for a given problem instance if, and
 * only if, all of the current state variables fall within the area covered.
 * Includes operations for copying, mutating, printing, etc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "data_structures.h"
#include "random.h"
#include "cl.h"
#include "cond_ellipsoid.h"

typedef struct COND_ELLIPSOID {
	double *center;
	double *radius;
	_Bool m;
	double *mu;
} COND_ELLIPSOID;

double cond_ellipsoid_dist(XCSF *xcsf, CL *c, double *x);

void cond_ellipsoid_init(XCSF *xcsf, CL *c)
{
	COND_ELLIPSOID *cond = malloc(sizeof(COND_ELLIPSOID));
	cond->center = malloc(sizeof(double) * xcsf->num_x_vars);
	cond->radius = malloc(sizeof(double) * xcsf->num_x_vars); 
	c->cond = cond;
	sam_init(xcsf, &cond->mu);
}

void cond_ellipsoid_free(XCSF *xcsf, CL *c)
{
	COND_ELLIPSOID *cond = c->cond;
	free(cond->center);
	free(cond->radius);
	sam_free(xcsf, cond->mu);
	free(c->cond);
}

double cond_ellipsoid_mu(XCSF *xcsf, CL *c, int m)
{
	(void)xcsf;
	COND_ELLIPSOID *cond = c->cond;
	return cond->mu[m];
}

void cond_ellipsoid_copy(XCSF *xcsf, CL *to, CL *from)
{
	COND_ELLIPSOID *to_cond = to->cond;
	COND_ELLIPSOID *from_cond = from->cond;
	memcpy(to_cond->center, from_cond->center, sizeof(double)*xcsf->num_x_vars);
	memcpy(to_cond->radius, from_cond->radius, sizeof(double)*xcsf->num_x_vars);
	sam_copy(xcsf, to_cond->mu, from_cond->mu);
}                             

void cond_ellipsoid_rand(XCSF *xcsf, CL *c)
{
	COND_ELLIPSOID *cond = c->cond;
	for(int i = 0; i < xcsf->num_x_vars; i++) {
		cond->center[i] = ((xcsf->MAX_CON - xcsf->MIN_CON) * drand()) + xcsf->MIN_CON;
		cond->radius[i] = (xcsf->MAX_CON - xcsf->MIN_CON) * drand() * 0.5;
	}
}

void cond_ellipsoid_cover(XCSF *xcsf, CL *c, double *x)
{
	COND_ELLIPSOID *cond = c->cond;
	for(int i = 0; i < xcsf->num_x_vars; i++) {
		cond->center[i] = x[i];
		cond->radius[i] = (xcsf->MAX_CON - xcsf->MIN_CON) * drand() * 0.5;
	}
}

_Bool cond_ellipsoid_match(XCSF *xcsf, CL *c, double *x)
{
	COND_ELLIPSOID *cond = c->cond;
	if(cond_ellipsoid_dist(xcsf, c, x) < 1.0) {
		cond->m = true;
	}
	else {
		cond->m = false;
	}
	return cond->m;
}
 
double cond_ellipsoid_dist(XCSF *xcsf, CL *c, double *x)
{
	COND_ELLIPSOID *cond = c->cond;
	double dist = 0.0;
	for(int i = 0; i < xcsf->num_x_vars; i++) {
		double d = (x[i] - cond->center[i]) / cond->radius[i];
		dist += d*d; // squared distance
	}
	return dist;
}
 
_Bool cond_ellipsoid_match_state(XCSF *xcsf, CL *c)
{
	(void)xcsf;
	COND_ELLIPSOID *cond = c->cond;
	return cond->m;
}

_Bool cond_ellipsoid_crossover(XCSF *xcsf, CL *c1, CL *c2) 
{
	COND_ELLIPSOID *cond1 = c1->cond;
	COND_ELLIPSOID *cond2 = c2->cond;
	_Bool changed = false;
	if(drand() < xcsf->P_CROSSOVER) {
		for(int i = 0; i < xcsf->num_x_vars; i++) {
			if(drand() < 0.5) {
				double tmp = cond1->center[i];
				cond1->center[i] = cond2->center[i];
				cond2->center[i] = tmp;
				changed = true;
			}
			if(drand() < 0.5) {
				double tmp = cond1->radius[i];
				cond1->radius[i] = cond2->radius[i];
				cond2->radius[i] = tmp;
				changed = true;
			}
		}
	}
	return changed;
}

_Bool cond_ellipsoid_mutate(XCSF *xcsf, CL *c)
{
	COND_ELLIPSOID *cond = c->cond;
	_Bool changed = false;
	// update mutation rates
	sam_adapt(xcsf, cond->mu);
    // apply mutation
	for(int i = 0; i < xcsf->num_x_vars; i++) {
		if(drand() < xcsf->P_MUTATION) {
			cond->center[i] += ((drand()*2.0)-1.0)*xcsf->S_MUTATION;
			if(cond->center[i] < xcsf->MIN_CON) {
				cond->center[i] = xcsf->MIN_CON;
			}
			else if(cond->center[i] > xcsf->MAX_CON) {
				cond->center[i] = xcsf->MAX_CON;
			}    
			changed = true;
		}
		if(drand() < xcsf->P_MUTATION) {
			cond->radius[i] += ((drand()*2.0)-1.0)*xcsf->S_MUTATION;
			changed = true;
		}
	}
	return changed;
}

_Bool cond_ellipsoid_general(XCSF *xcsf, CL *c1, CL *c2)
{
	// returns whether cond1 is more general than cond2
	COND_ELLIPSOID *cond1 = c1->cond;
	COND_ELLIPSOID *cond2 = c2->cond;
	for(int i = 0; i < xcsf->num_x_vars; i++) {
		double l1 = cond1->center[i] - cond1->radius[i];
		double l2 = cond2->center[i] - cond2->radius[i];
		double u1 = cond1->center[i] + cond1->radius[i];
		double u2 = cond2->center[i] + cond2->radius[i];
		if(l1 > l2 || u1 < u2) {
			return false;
		}
	}
	return true;
}  

void cond_ellipsoid_print(XCSF *xcsf, CL *c)
{
	COND_ELLIPSOID *cond = c->cond;
	printf("ellipsoid:");
	for(int i = 0; i < xcsf->num_x_vars; i++) {
		printf(" (%5f, ", cond->center[i]);
		printf("%5f)", cond->radius[i]);
	}
	printf("\n");
}
