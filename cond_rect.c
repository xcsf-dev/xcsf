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

#if CON == 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

void cond_bounds(double *a, double *b);

void cond_init(CL *c)
{
	c->cond.interval_length = state_length*2;
	c->cond.interval = malloc(sizeof(double) * c->cond.interval_length);
#ifdef SAM
	sam_init(&c->cond.mu);
#endif
}

void cond_free(CL *c)
{
	free(c->cond.interval);
#ifdef SAM
	sam_free(c->cond.mu);
#endif
}

void cond_copy(CL *to, CL *from)
{
	memcpy(to->cond.interval, from->cond.interval, sizeof(double) * to->cond.interval_length);
#ifdef SAM
	memcpy(to->cond.mu, from->cond.mu, sizeof(double)*NUM_MU);
#endif
}                             

void cond_rand(CL *c)
{
	for(int i = 0; i < c->cond.interval_length; i+=2) {
		c->cond.interval[i] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		c->cond.interval[i+1] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		cond_bounds(&c->cond.interval[i], &c->cond.interval[i+1]);
	}
}

void cond_cover(CL *c, double *state)
{
	// generate a condition that matches the state
	for(int i = 0; i < c->cond.interval_length; i+=2) {
		c->cond.interval[i] = state[i/2] - (S_MUTATION*drand());
		c->cond.interval[i+1] = state[i/2] + (S_MUTATION*drand());
		cond_bounds(&c->cond.interval[i], &c->cond.interval[i+1]);
	}
}

void cond_bounds(double *a, double *b)
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

_Bool cond_match(CL *c, double *state)
{
	// return whether the condition matches the state
	for(int i = 0; i < c->cond.interval_length; i+=2) {
		if(state[i/2] < c->cond.interval[i] || state[i/2] > c->cond.interval[i+1]) {
			c->cond.m = false;
			return false;
		}
	}
	c->cond.m = true;
	return true;
}

_Bool cond_crossover(CL *c1, CL *c2) 
{
	// two point crossover
	_Bool changed = false;
	int length = c1->cond.interval_length;
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
		memcpy(cl1, c1->cond.interval, sizeof(double)*length);
		memcpy(cl2, c2->cond.interval, sizeof(double)*length);
		for(int i = p1; i < p2; i++) { 
			if(cl1[i] != cl2[i]) {
				changed = true;
				double help = c1->cond.interval[i];
				c1->cond.interval[i] = cl2[i];
				c2->cond.interval[i] = help;
			}
		}
		if(changed) {
			memcpy(c1->cond.interval, cl1, sizeof(double)*length);
			memcpy(c2->cond.interval, cl2, sizeof(double)*length);
		}
	}
	return changed;
}

_Bool cond_mutate(CL *c)
{
	_Bool mod = false;
	double step = S_MUTATION;
#ifdef SAM
	sam_adapt(c->cond.mu);
	if(NUM_MU > 0) {
		P_MUTATION = c->cond.mu[0];
		if(NUM_MU > 1)
			step = c->cond.mu[1];
	}
#endif
	for(int i = 0; i < c->cond.interval_length; i+=2) {
		if(drand() < P_MUTATION) {
			c->cond.interval[i] += ((drand()*2.0)-1.0)*step;
			mod = true;
		}
		if(drand() < P_MUTATION) {
			c->cond.interval[i+1] += ((drand()*2.0)-1.0)*step;
			mod = true;
		}
		cond_bounds(&c->cond.interval[i], &c->cond.interval[i+1]);
	}
	return mod;
}

_Bool cond_subsumes(CL *c1, CL *c2)
{
	// returns whether c1 subsumes c2
	for(int i = 0; i < c1->cond.interval_length; i+=2) {
		if(c1->cond.interval[i] > c2->cond.interval[i] 
				|| c1->cond.interval[i+1] < c2->cond.interval[i+1]) {
			return false;
		}
	}
	return true;
}

_Bool cond_general(CL *c1, CL *c2)
{
	// returns whether cond1 is more general than cond2
	double gen1 = 0.0, gen2 = 0.0, max = 0.0;
	for(int i = 0; i < state_length; i++)
		max += MAX_CON - MIN_CON + 1.0;
	for(int i = 0; i < c1->cond.interval_length; i+=2) {
		gen1 += c1->cond.interval[i+1] - c1->cond.interval[i] + 1.0;
		gen2 += c2->cond.interval[i+1] - c2->cond.interval[i] + 1.0;
	}
	if(gen1/max > gen2/max)
		return false;
	else
		return true;
}  

void cond_print(CL *c)
{
	printf("intervals:");
	for(int i = 0; i < c->cond.interval_length; i+=2) {
		printf(" (%5f, ", c->cond.interval[i]);
		printf("%5f)", c->cond.interval[i+1]);
	}
	printf("\n");
}
#endif
