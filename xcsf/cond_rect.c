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
 * The hyperrectangle classifier condition module.
 *
 * Provides functionality to create real-valued hyperrectangle (interval)
 * conditions whereby a classifier matches for a given problem instance if, and
 * only if, all of the current state variables fall within all {lower, upper}
 * intervals. Includes operations for copying, mutating, printing, etc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"
#include "cond_rect.h"

void cond_rect_bounds(double *a, double *b);
  
typedef struct COND_RECT {
	double *interval;
	int interval_length;
	_Bool m;
	double *mu;
} COND_RECT;
 
void cond_rect_init(CL *c)
{
	COND_RECT *cond = malloc(sizeof(COND_RECT));
	cond->interval_length = num_x_vars*2;
	cond->interval = malloc(sizeof(double) * cond->interval_length);
	c->cond = cond;
	sam_init(&cond->mu);
}

void cond_rect_free(CL *c)
{
	COND_RECT *cond = c->cond;
	free(cond->interval);
	sam_free(cond->mu);
	free(c->cond);
}

double cond_rect_mu(CL *c, int m)
{
	COND_RECT *cond = c->cond;
	return cond->mu[m];
}

void cond_rect_copy(CL *to, CL *from)
{
	COND_RECT *to_cond = to->cond;
	COND_RECT *from_cond = from->cond;
	memcpy(to_cond->interval, from_cond->interval, sizeof(double)*to_cond->interval_length);
	sam_copy(to_cond->mu, from_cond->mu);
}                             

void cond_rect_rand(CL *c)
{
	COND_RECT *cond = c->cond;
	for(int i = 0; i < cond->interval_length; i+=2) {
		cond->interval[i] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		cond->interval[i+1] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		cond_rect_bounds(&cond->interval[i], &cond->interval[i+1]);
	}
}

void cond_rect_cover(CL *c, double *x)
{
	COND_RECT *cond = c->cond;
	// generate a condition that matches the state
	for(int i = 0; i < cond->interval_length; i+=2) {
		cond->interval[i] = x[i/2] - (S_MUTATION*drand());
		cond->interval[i+1] = x[i/2] + (S_MUTATION*drand());
		cond_rect_bounds(&cond->interval[i], &cond->interval[i+1]);
	}
}

void cond_rect_bounds(double *a, double *b)
{
	// lower and upper limits
	if(*a < MIN_CON)
		*a = MIN_CON;
	else if(*a > MAX_CON)
		*a = MAX_CON;
	if(*b < MIN_CON)
		*b = MIN_CON;
	else if(*b > MAX_CON)
		*b = MAX_CON;
	// order
	if(*a > *b) {
		double tmp = *a;
		*a = *b;
		*b = tmp;
	}                              
}

_Bool cond_rect_match(CL *c, double *x)
{
	COND_RECT *cond = c->cond;
	// return whether the condition matches the state
	for(int i = 0; i < cond->interval_length; i+=2) {
		if(x[i/2] < cond->interval[i] || x[i/2] > cond->interval[i+1]) {
			cond->m = false;
			return false;
		}
	}
	cond->m = true;
	return true;
}

_Bool cond_rect_match_state(CL *c)
{
	COND_RECT *cond = c->cond;
	return cond->m;
}

_Bool cond_rect_crossover(CL *c1, CL *c2) 
{
	COND_RECT *cond1 = c1->cond;
	COND_RECT *cond2 = c2->cond;
	// two point crossover
	_Bool changed = false;
	int length = cond1->interval_length;
	if(drand() < P_CROSSOVER) {
		int p1 = irand(0, length);
		int p2 = irand(0, length)+1;
		if(p1 > p2) {
			int help = p1;
			p1 = p2;
			p2 = help;
		}
		else if(p1 == p2) {
			p2++;
		}
		double cl1[length];
		double cl2[length];
		memcpy(cl1, cond1->interval, sizeof(double)*length);
		memcpy(cl2, cond2->interval, sizeof(double)*length);
		for(int i = p1; i < p2; i++) { 
			if(cl1[i] != cl2[i]) {
				changed = true;
				double help = cond1->interval[i];
				cond1->interval[i] = cl2[i];
				cond2->interval[i] = help;
			}
		}
		if(changed) {
			memcpy(cond1->interval, cl1, sizeof(double)*length);
			memcpy(cond2->interval, cl2, sizeof(double)*length);
		}
	}
	return changed;
}

_Bool cond_rect_mutate(CL *c)
{
	COND_RECT *cond = c->cond;
	_Bool mod = false;
	double step = S_MUTATION;
	if(NUM_SAM > 0) {
		sam_adapt(cond->mu);
		P_MUTATION = cond->mu[0];
		if(NUM_SAM > 1)
			step = cond->mu[1];
	}
	for(int i = 0; i < cond->interval_length; i+=2) {
		if(drand() < P_MUTATION) {
			cond->interval[i] += ((drand()*2.0)-1.0)*step;
			mod = true;
		}
		if(drand() < P_MUTATION) {
			cond->interval[i+1] += ((drand()*2.0)-1.0)*step;
			mod = true;
		}
		cond_rect_bounds(&cond->interval[i], &cond->interval[i+1]);
	}
	return mod;
}

_Bool cond_rect_subsumes(CL *c1, CL *c2)
{
	COND_RECT *cond1 = c1->cond;
	COND_RECT *cond2 = c2->cond;
	// returns whether c1 subsumes c2
	for(int i = 0; i < cond1->interval_length; i+=2) {
		if(cond1->interval[i] > cond2->interval[i] 
				|| cond1->interval[i+1] < cond2->interval[i+1]) {
			return false;
		}
	}
	return true;
}

_Bool cond_rect_general(CL *c1, CL *c2)
{
	// returns whether cond1 is more general than cond2
	COND_RECT *cond1 = c1->cond;
	COND_RECT *cond2 = c2->cond;
	double gen1 = 0.0, gen2 = 0.0, max = 0.0;
	for(int i = 0; i < num_x_vars; i++)
		max += MAX_CON - MIN_CON + 1.0;
	for(int i = 0; i < cond1->interval_length; i+=2) {
		gen1 += cond1->interval[i+1] - cond1->interval[i] + 1.0;
		gen2 += cond2->interval[i+1] - cond2->interval[i] + 1.0;
	}
	if(gen1/max > gen2/max) {
		return false;
	}
	else {
		return true;
	}
}  

void cond_rect_print(CL *c)
{
	COND_RECT *cond = c->cond;
	printf("intervals:");
	for(int i = 0; i < cond->interval_length; i+=2) {
		printf(" (%5f, ", cond->interval[i]);
		printf("%5f)", cond->interval[i+1]);
	}
	printf("\n");
}
