/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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

void cond_init(COND *cond)
{
	cond->interval_length = state_length*2;
	cond->interval = malloc(sizeof(double) * cond->interval_length);
#ifdef SAM
	sam_init(&cond->mu);
#endif
}

void cond_free(COND *cond)
{
	free(cond->interval);
#ifdef SAM
	sam_free(cond->mu);
#endif
}

void cond_copy(COND *to, COND *from)
{
	memcpy(to->interval, from->interval, sizeof(double) * to->interval_length);
#ifdef SAM
	memcpy(to->mu, from->mu, sizeof(double)*NUM_MU);
#endif
}                             

void cond_rand(COND *cond)
{
	for(int i = 0; i < cond->interval_length; i+=2) {
		cond->interval[i] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		cond->interval[i+1] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		cond_bounds(&cond->interval[i], &cond->interval[i+1]);
	}
}

void cond_cover(COND *cond, double *state)
{
	// generate a condition that matches the state
	for(int i = 0; i < cond->interval_length; i+=2) {
		cond->interval[i] = state[i/2] - (S_MUTATION*drand());
		cond->interval[i+1] = state[i/2] + (S_MUTATION*drand());
		cond_bounds(&cond->interval[i], &cond->interval[i+1]);
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

_Bool cond_match(COND *cond, double *state)
{
	// return whether the condition matches the state
	for(int i = 0; i < cond->interval_length; i+=2) {
		if(state[i/2] < cond->interval[i] || state[i/2] > cond->interval[i+1]) {
			cond->m = false;
			return false;
		}
	}
	cond->m = true;
	return true;
}

_Bool cond_crossover(COND *cond1, COND *cond2) 
{
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
		double c1[length];
		double c2[length];
		memcpy(c1, cond1->interval, sizeof(double)*length);
		memcpy(c2, cond2->interval, sizeof(double)*length);
		for(int i = p1; i < p2; i++) { 
			if(c1[i] != c2[i]) {
				changed = true;
				double help = cond1->interval[i];
				cond1->interval[i] = c2[i];
				cond2->interval[i] = help;
			}
		}
		if(changed) {
			memcpy(cond1->interval, c1, sizeof(double)*length);
			memcpy(cond2->interval, c2, sizeof(double)*length);
		}
	}
	return changed;
}

_Bool cond_mutate(COND *cond)
{
	_Bool mod = false;
	double step = S_MUTATION;
#ifdef SAM
	sam_adapt(cond->mu);
	if(NUM_MU > 0) {
		P_MUTATION = cond->mu[0];
		if(NUM_MU > 1)
			step = cond->mu[1];
	}
#endif
	for(int i = 0; i < cond->interval_length; i+=2) {
		if(drand() < P_MUTATION) {
			cond->interval[i] += ((drand()*2.0)-1.0)*step;
			mod = true;
		}
		if(drand() < P_MUTATION) {
			cond->interval[i+1] += ((drand()*2.0)-1.0)*step;
			mod = true;
		}
		cond_bounds(&cond->interval[i], &cond->interval[i+1]);
	}
	return mod;
}

_Bool cond_subsumes(COND *cond1, COND *cond2)
{
	// returns whether cond1 subsumes cond2
	for(int i = 0; i < cond1->interval_length; i+=2) {
		if(cond1->interval[i] > cond2->interval[i] 
				|| cond1->interval[i+1] < cond2->interval[i+1]) {
			return false;
		}
	}
	return true;
}

_Bool cond_general(COND *cond1, COND *cond2)
{
	// returns whether cond1 is more general than cond2
	double gen1 = 0.0, gen2 = 0.0, max = 0.0;
	for(int i = 0; i < state_length; i++)
		max += MAX_CON - MIN_CON + 1.0;
	for(int i = 0; i < cond1->interval_length; i+=2) {
		gen1 += cond1->interval[i+1] - cond1->interval[i] + 1.0;
		gen2 += cond2->interval[i+1] - cond2->interval[i] + 1.0;
	}
	if(gen1/max > gen2/max)
		return false;
	else
		return true;
}  

void cond_print(COND *cond)
{
	printf("intervals:");
	for(int i = 0; i < cond->interval_length; i+=2) {
		printf(" (%5f, ", cond->interval[i]);
		printf("%5f)", cond->interval[i+1]);
	}
	printf("\n");
}
#endif
