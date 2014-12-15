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
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "cons.h"
#include "random.h"
#include "cl.h"
#ifdef NEURAL_CONDITIONS
#include "bpn.h"
#endif

#ifdef SELF_ADAPT_MUTATION
void adapt_mut(CL *c);
double gasdev(CL *c, int m);
#endif

#ifndef NEURAL_CONDITIONS
void bounds(double *a, double *b);
#endif

void init_cl(CL *c, int size, int time)
{
	c->fit = INIT_FITNESS;
	c->err = INIT_ERROR;
	c->num = 1;
	c->exp = 0;
	c->size = size;
	c->time = time;
	c->weights_length = (state_length*XCSF_EXPONENT)+1;
#ifdef NEURAL_CONDITIONS
#define NUM_OUTPUT 1 // only one output required for matching
	c->con_length = ((state_length+1)*NUM_HIDDEN_NEURONS)
		+((NUM_HIDDEN_NEURONS+1)*NUM_OUTPUT);
#else
	c->con_length = state_length*2;
#endif
	c->con = malloc(sizeof(double) * c->con_length);
	c->weights = malloc(sizeof(double) * c->weights_length);
	for(int i = 0; i < c->weights_length; i++)
		c->weights[i] = 0.0;
#ifdef SELF_ADAPT_MUTATION
	c->mu = malloc(sizeof(double)*NUM_MU);
	c->iset = malloc(sizeof(int)*NUM_MU);
	c->gset = malloc(sizeof(double)*NUM_MU);
	for(int i = 0; i < NUM_MU; i++) {
		c->mu[i] = drand();
		c->iset[i] = 0;
		c->gset[i] = 0.0;
	}
#endif
}

void copy_cl(CL *to, CL *from)
{
	init_cl(to, from->size, from->time);
	to->weights_length = from->weights_length;
	memcpy(to->weights, from->weights, sizeof(double)*from->weights_length);
	to->con_length = from->con_length;
	memcpy(to->con, from->con, sizeof(double)*from->con_length);
#ifdef SELF_ADAPT_MUTATION
	memcpy(to->mu, from->mu, sizeof(double)*NUM_MU);
	memcpy(to->gset, from->gset, sizeof(double)*NUM_MU);
	memcpy(to->iset, from->iset, sizeof(int)*NUM_MU);
#endif
}

#ifdef NEURAL_CONDITIONS
void rand_con(CL *c)
{
	for(int i = 0; i < c->con_length; i++)
		c->con[i] = (drand()*2.0)-1.0;
}

void match_con(CL *c, double *state)
{
	// generates random weights until the network matches for input state
	do {
		for(int i = 0; i < c->con_length; i++)
			c->con[i] = (drand()*2.0)-1.0;
	} while(!match(c, state));
}

_Bool match(CL *c, double *state)
{
	// classifier matches if the first output neuron > 0.5
	neural_set_weights(c->con);
	neural_propagate(state);
	if(neural_output(0) > 0.5)
		return true;
	return false;
}

_Bool mutate(CL *c)
{
	double mod = false;
	double step = S_MUTATION;
#ifdef SELF_ADAPT_MUTATION
	adapt_mut(c);
	if(NUM_MU > 0) {
		P_MUTATION = c->mu[0];
		if(NUM_MU > 1)
			step = c->mu[1];
	}
#endif
	for(int i = 0; i < c->con_length; i++) {
		if(drand() < P_MUTATION) {
			c->con[i] += ((drand()*2.0)-1.0)*step;
			if(c->con[i] > 1.0)
				c->con[i] = 1.0;
			else if(c->con[i] < -1.0)
				c->con[i] = -1.0;
			mod = true;
		}
	}
	return mod;
}

_Bool subsumes(CL *c1, CL *c2)
{
	return false;
}

_Bool general(CL *c1, CL *c2)
{
	return false;
}
#else
void rand_con(CL *c)
{
	for(int i = 0; i < state_length+1; i+=2) {
		c->con[i] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		c->con[i+1] = ((MAX_CON-MIN_CON)*drand())+MIN_CON;
		bounds(&(c->con[i]), &(c->con[i+1]));
	}
}

void match_con(CL *c, double *state)
{
	// generate a condition that matches the state
	for(int i = 0; i < state_length*2; i+=2) {
		c->con[i] = state[i/2] - (S_MUTATION*2.0);
		c->con[i+1] = state[i/2] + (S_MUTATION*2.0);
		bounds(&(c->con[i]), &(c->con[i+1]));
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
	// return whether the condition matches the state
	for(int i = 0; i < state_length*2; i+=2) {
		if(state[i/2] < c->con[i] || state[i/2] > c->con[i+1])
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
		double cond1[state_length*2];
		double cond2[state_length*2];
		memcpy(cond1, c1->con, sizeof(double)*state_length*2);
		memcpy(cond2, c2->con, sizeof(double)*state_length*2);
		for(int i = p1; i < p2; i++) { 
			if(cond1[i] != cond2[i]) {
				changed = true;
				double help = c1->con[i];
				c1->con[i] = cond2[i];
				c2->con[i] = help;
			}
		}
		if(changed) {
			memcpy(c1->con, cond1, sizeof(double)*state_length*2);
			memcpy(c2->con, cond2, sizeof(double)*state_length*2);
		}
	}
	return changed;
}

_Bool mutate(CL *c)
{
	double step = S_MUTATION;
#ifdef SELF_ADAPT_MUTATION
	adapt_mut(c);
	if(NUM_MU > 0) {
		P_MUTATION = c->mu[0];
		if(NUM_MU > 1)
			step = c->mu[1];
	}

#endif
	for(int i = 0; i < state_length*2; i+=2) {
		if(drand() < P_MUTATION)
			c->con[i] += ((drand()*2.0)-1.0)*step;
		if(drand() < P_MUTATION)
			c->con[i+1] += ((drand()*2.0)-1.0)*step;

		// bounds
		bounds(&(c->con[i]), &(c->con[i+1]));

	}
	return true;
}

_Bool subsumes(CL *c1, CL *c2)
{
	// returns whether c1 subsumes c2
	if(subsumer(c1)) {
		for(int i = 0; i < state_length*2; i+=2) {
			if(c1->con[i] > c2->con[i] || c1->con[i+1] < c2->con[i+1])
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
		gen1 += c1->con[i+1] - c1->con[i] + 1.0;
		gen2 += c2->con[i+1] - c2->con[i] + 1.0;
	}
	if(gen1/max > gen2/max)
		return false;
	else
		return true;
}
#endif

_Bool subsumer(CL *c)
{
	if(c->exp > THETA_SUB && c->err < EPS_0)
		return true;
	else
		return false;
}

double del_vote(CL *c, double avg_fit)
{
	if(c->fit / c->num >= DELTA * avg_fit || c->exp < THETA_DEL)
		return c->size * c->num;
	return c->size * c->num * avg_fit / (c->fit / c->num); 
}

void update_pre(CL *c, double p, double *state)
{
	double error = p - compute_pre(c, state);
	double norm = XCSF_X0 * XCSF_X0;
	for(int i = 0; i < state_length; i++)
		norm += state[i] * state[i];
	double correction = (XCSF_ETA * error) / norm;
	c->weights[0] += XCSF_X0 * correction;
	for(int i = 0; i < c->weights_length-1; i+=XCSF_EXPONENT)
		for(int j = 0; j < XCSF_EXPONENT; j++)
			c->weights[i+j+1] += correction * pow(state[i/XCSF_EXPONENT], j+1);
}

double compute_pre(CL *c, double *state)
{
	double pre = XCSF_X0 * c->weights[0];
	for(int i = 0; i < c->weights_length-1; i+=XCSF_EXPONENT)
		for(int j = 0; j < XCSF_EXPONENT; j++)
			pre += pow(state[i/XCSF_EXPONENT], j+1) * c->weights[i+j+1];
	return pre;
} 

double update_err(CL *c, double p, double *state)
{
	double pre = compute_pre(c, state);
	if(c->exp < 1.0/BETA) 
		c->err = (c->err * (c->exp-1.0) + fabs(p - pre)) / (double)c->exp;
	else
		c->err += BETA * (fabs(p - pre) - c->err);
	return c->err * c->num;
}

double acc(CL *c)
{
	if(c->err <= EPS_0)
		return 1.0;
	else
		return ALPHA * pow(c->err / EPS_0, -NU);
}

void update_fit(CL *c, double acc_sum, double acc)
{
	c->fit += BETA * ((acc * c->num) / acc_sum - c->fit);
}

double update_size(CL *c, double num_sum)
{
	if(c->exp < 1.0/BETA)
		c->size = (c->size * (c->exp-1.0) + num_sum) / (double)c->exp; 
	else
		c->size += BETA * (num_sum - c->size);
	return c->size * c->num;
}

void free_cl(CL *c)
{
	free(c->con);
	free(c->weights);
#ifdef SELF_ADAPT_MUTATION
	free(c->mu);
	free(c->iset);
	free(c->gset);
#endif
	free(c);
}

void print_cl(CL *c)
{
#ifdef NEURAL_CONDITIONS
	printf("neural weights:");
	for(int i = 0; i < c->con_length; i++)
		printf(" %5f, ", c->con[i]);

#else
	printf("intervals:");
	for(int i = 0; i < c->con_length; i+=2) {
		printf(" (%5f, ", c->con[i]);
		printf("%5f)", c->con[i+1]);
	}
#endif
	printf("\n %f %f %d %d %f %d\n",
			c->err, c->fit, c->num, c->exp, c->size, c->time);
	printf("weights: ");
	for(int i = 0; i < c->weights_length; i++)
		printf("%f, ", c->weights[i]);
	printf("\n");
}

#ifdef SELF_ADAPT_MUTATION
void adapt_mut(CL *c)
{
	for(int i = 0; i < NUM_MU; i++) {
		c->mu[i] *= exp(gasdev(c,i));
		if(c->mu[i] < muEPS_0)
			c->mu[i] = muEPS_0;
		else if(c->mu[i] > 1.0)
			c->mu[i] = 1.0;
	}
}

double gasdev(CL *c, int m)
{
	// from numerical recipes in c
	double fac, rsq, v1, v2;
	if(c->iset[m] == 0) {
		do {
			v1 = (drand()*2.0)-1.0;
			v2 = (drand()*2.0)-1.0;
			rsq = (v1*v1)+(v2*v2);
		}
		while(rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0*log(rsq)/rsq);
		c->gset[m] = v1*fac;
		c->iset[m] = 1;
		return v2*fac;
	}
	else {
		c->iset[m] = 0;
		return c->gset[m];
	}
}
#endif
