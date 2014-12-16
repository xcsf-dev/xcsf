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

#ifndef NEURAL_CONDITIONS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

void bounds(double *a, double *b);

void cond_init(CL *c)
{
	c->cond_length = state_length*2;
	c->cond = malloc(sizeof(double) * c->cond_length);
}

void cond_free(CL *c)
{
	free(c->cond);
}

void cond_copy(CL *to, CL *from)
{
	to->cond_length = from->cond_length;
	memcpy(to->cond, from->cond, sizeof(double)*from->cond_length);

}                             

void cond_rand(CL *c)
{
	for(int i = 0; i < state_length+1; i+=2) {
		c->cond[i] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		c->cond[i+1] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		bounds(&(c->cond[i]), &(c->cond[i+1]));
	}
}

void cond_match(CL *c, double *state)
{
	// generate a conddition that matches the state
	for(int i = 0; i < state_length*2; i+=2) {
		c->cond[i] = state[i/2] - (S_MUTATION*2.0);
		c->cond[i+1] = state[i/2] + (S_MUTATION*2.0);
		bounds(&(c->cond[i]), &(c->cond[i+1]));
	}
}

void bounds(double *a, double *b)
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

_Bool match(CL *c, double *state)
{
	// return whether the conddition matches the state
	for(int i = 0; i < state_length*2; i+=2) {
		if(state[i/2] < c->cond[i] || state[i/2] > c->cond[i+1])
			return false;
	}
	return true;
}

_Bool two_pt_cross(CL *c1, CL *c2) 
{
	_Bool changed = false;
	if(drand() < P_CROSSOVER) {
		int p1 = irand(0, state_length*2);
		int p2 = irand(0, state_length*2)+1;
		if(p1 > p2) {
			int help = p1;
			p1 = p2;
			p2 = help;
		}
		else if(p1 == p2) {
			p2++;
		}
		double condd1[state_length*2];
		double condd2[state_length*2];
		memcpy(condd1, c1->cond, sizeof(double)*state_length*2);
		memcpy(condd2, c2->cond, sizeof(double)*state_length*2);
		for(int i = p1; i < p2; i++) { 
			if(condd1[i] != condd2[i]) {
				changed = true;
				double help = c1->cond[i];
				c1->cond[i] = condd2[i];
				c2->cond[i] = help;
			}
		}
		if(changed) {
			memcpy(c1->cond, condd1, sizeof(double)*state_length*2);
			memcpy(c2->cond, condd2, sizeof(double)*state_length*2);
		}
	}
	return changed;
}

_Bool mutate(CL *c)
{
	double step = S_MUTATION;
#ifdef SELF_ADAPT_MUTATION
	sam_adapt(c);
	if(NUM_MU > 0) {
		P_MUTATION = c->mu[0];
		if(NUM_MU > 1)
			step = c->mu[1];
	}

#endif
	for(int i = 0; i < state_length*2; i+=2) {
		if(drand() < P_MUTATION)
			c->cond[i] += ((drand()*2.0)-1.0)*step;
		if(drand() < P_MUTATION)
			c->cond[i+1] += ((drand()*2.0)-1.0)*step;

		// bounds
		bounds(&(c->cond[i]), &(c->cond[i+1]));

	}
	return true;
}

_Bool subsumes(CL *c1, CL *c2)
{
	// returns whether c1 subsumes c2
	if(cl_subsumer(c1)) {
		for(int i = 0; i < state_length*2; i+=2) {
			if(c1->cond[i] > c2->cond[i] || c1->cond[i+1] < c2->cond[i+1])
				return false;
		}
		return true;
	}
	return false;
}

_Bool general(CL *c1, CL *c2)
{
	// returns whether c1 is more general than c2
	double gen1 = 0.0, gen2 = 0.0, max = 0.0;
	for(int i = 0; i < state_length; i++)
		max += MAX_CON - MIN_CON + 1.0;
	for(int i = 0; i < state_length*2; i+=2) {
		gen1 += c1->cond[i+1] - c1->cond[i] + 1.0;
		gen2 += c2->cond[i+1] - c2->cond[i] + 1.0;
	}
	if(gen1/max > gen2/max)
		return false;
	else
		return true;
}  

void cond_print(CL *c)
{
	printf("intervals:");
	for(int i = 0; i < c->cond_length; i+=2) {
		printf(" (%5f, ", c->cond[i]);
		printf("%5f)", c->cond[i+1]);
	}
	printf("\n");
}
#endif
