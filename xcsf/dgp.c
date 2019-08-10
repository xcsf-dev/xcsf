/*
 * Copyright (C) 2016--2019 Richard Preen <rpreen@gmail.com>
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "data_structures.h"
#include "random.h"
#include "dgp.h"

#define NUM_FUNC 7 // number of node functions available

char node_symbol(int func);
void node_update(XCSF *xcsf, double *state, int func, double input);

void graph_init(XCSF *xcsf, GRAPH *dgp, int n)
{
	dgp->avgk = 0.0;
	dgp->t = 0;
	dgp->n = n;
	dgp->state = malloc(sizeof(double)*dgp->n);
	dgp->initial_state = malloc(sizeof(double)*dgp->n);
	dgp->function = malloc(sizeof(int)*dgp->n);
	dgp->connectivity = malloc(sizeof(int)*dgp->n*xcsf->MAX_K);
	graph_rand(xcsf, dgp);
}

void graph_copy(XCSF *xcsf, GRAPH *to, GRAPH *from)
{ 	
	to->avgk = from->avgk;
	to->t = from->t;
	to->n = from->n;
	memcpy(to->state, from->state, sizeof(double)*from->n);
	memcpy(to->initial_state, from->initial_state, sizeof(double)*from->n);
	memcpy(to->function, from->function, sizeof(int)*from->n);
	memcpy(to->connectivity, from->connectivity, sizeof(int)*from->n*xcsf->MAX_K);
}

double graph_output(XCSF *xcsf, GRAPH *dgp, int i)
{
	(void)xcsf;
	return dgp->state[i];
}

void graph_reset(XCSF *xcsf, GRAPH *dgp)
{
	(void)xcsf;
	for(int i = 0; i < dgp->n; i++) {
		dgp->state[i] = dgp->initial_state[i];
	}
}

void graph_rand(XCSF *xcsf, GRAPH *dgp)
{
	dgp->t = irand(1,xcsf->MAX_T);
	for(int i = 0; i < dgp->n; i++) {
		dgp->function[i] = irand(0, NUM_FUNC);
		dgp->initial_state[i] = ((xcsf->MAX_CON-xcsf->MIN_CON)*drand())+xcsf->MIN_CON;
		dgp->state[i] = ((xcsf->MAX_CON-xcsf->MIN_CON)*drand())+xcsf->MIN_CON;
	}
	for(int i = 0; i < dgp->n * xcsf->MAX_K; i++) {
		if(drand() < 0.5) {
			dgp->connectivity[i] = 0; // inert
		}
		else {
			// other nodes within the graph
			if(drand() < 0.5) {
				dgp->connectivity[i] = irand(1,dgp->n+1);
			}
			// external inputs
			else {
				dgp->connectivity[i] = -(irand(1,xcsf->num_x_vars+1));
			}
		}
	}  
	// set avg k
	dgp->avgk = 0;
	for(int i = 0; i < dgp->n*xcsf->MAX_K; i++) {
		if(dgp->connectivity[i] != 0) {
			dgp->avgk++;
		}
	}
	dgp->avgk /= (double)dgp->n;
}

void graph_update(XCSF *xcsf, GRAPH *dgp, double *inputs)
{
	if(xcsf->RESET_STATES) {
		graph_reset(xcsf, dgp);
	}
	for(int t = 0; t < dgp->t; t++) {
		// synchronously update each node
		for(int i = 0; i < dgp->n; i++) {
			// each connection
			for(int k = 0; k < xcsf->MAX_K; k++) {
				int c = dgp->connectivity[(i*xcsf->MAX_K)+k];
				// inert
				if(c == 0) {
					continue;
				}
				// another node within the graph
				else if(c > 0) {
					node_update(xcsf, &dgp->state[i], dgp->function[i], dgp->state[c-1]);
				}
				// external input
				else {
					node_update(xcsf, &dgp->state[i], dgp->function[i], inputs[abs(c)-1]);
				}
			}
		}
	}
}

void graph_print(XCSF *xcsf, GRAPH *dgp)
{
	printf("Graph: N=%d; T=%d\n", dgp->n, dgp->t);
	for(int i = 0; i < dgp->n; i++) {
		printf("Node %d: %c state=%f init_state=%f con=[", 
				i, node_symbol(dgp->function[i]), 
				dgp->state[i], dgp->initial_state[i]);
		printf("%d", dgp->connectivity[0]);
		for(int j = 1; j < xcsf->MAX_K; j++) {
			printf(", %d", dgp->connectivity[i]);
		}
		printf("]\n");
	}
	(void)xcsf;
}

void graph_free(XCSF *xcsf, GRAPH *dgp)
{
	free(dgp->connectivity);
	free(dgp->state);
	free(dgp->initial_state);
	free(dgp->function);
	(void)xcsf;
}

_Bool graph_mutate(XCSF *xcsf, GRAPH *dgp, double rate)
{
	_Bool fmodified = false;
	_Bool cmodified = false;
	_Bool tmodified = false;

	for(int i = 0; i < dgp->n; i++) {
		// mutate function
		if(drand() < rate) {
			int old = dgp->function[i];
			dgp->function[i] = irand(0, NUM_FUNC);
			if(old != dgp->function[i]) {
				fmodified = true;
			}              
		}
		// mutate connectivity map
		for(int j = 0; j < xcsf->MAX_K; j++) {
			int idx = (i*xcsf->MAX_K)+j;
			if(drand() < rate) {
				int old = dgp->connectivity[idx];
				// inert
				if(drand() < 0.1) {
					dgp->connectivity[idx] = 0;
				}
				// external connection
				else if(drand() < 0.2) {
					dgp->connectivity[idx] = -(irand(1,xcsf->num_x_vars+1));
				}
				// another node
				else {
					dgp->connectivity[idx] = irand(1,dgp->n+1);
				}
				if(old != dgp->connectivity[idx]) {
					cmodified = true;
				}
			}
		}   
	}

	// mutate T
	if(drand() < rate) {
		int t = dgp->t;
		if(drand() < 0.5) {
			if(dgp->t > 1) {
				(dgp->t)--;
			}
		}
		else {
			if(dgp->t < xcsf->MAX_T) {
				(dgp->t)++;
			}
		}
		if(t != dgp->t) {
			tmodified = true;
		}
	}

	// refresh k
	if(cmodified) {
		dgp->avgk = 0;
		for(int i = 0; i < dgp->n*xcsf->MAX_K; i++) {
			if(dgp->connectivity[i] != 0) {
				dgp->avgk++;
			}
		}
		dgp->avgk /= (double)dgp->n;
	}            

	if(fmodified || cmodified || tmodified) {
		return true;
	}
	else {
		return false;
	}
}

double graph_avg_k(XCSF *xcsf, GRAPH *dgp)
{
	(void)xcsf;
	return dgp->avgk;
}

char node_symbol(int func)
{
	switch(func) {
		case 0: return '+';
		case 1: return '-';
		case 2: return '*';
		case 3: return '/';
		case 4: return 'S';
		case 5: return 'C';
		case 6: return 'T';
		default: return ' ';
	}
}

void node_update(XCSF *xcsf, double *state, int func, double input)
{
	switch(func) {
		case 0: *state += input; break;
		case 1: *state -= input; break;
		case 2: *state *= input; break;
		case 3: if(input != 0.0) *state /= input; break;
		case 4: *state = sin(input); break;
		case 5: *state = cos(input); break;
		case 6: *state = tanh(input); break;
		default: break;
	}
	if(*state > xcsf->MAX_CON) {
		*state = xcsf->MAX_CON;
	}
	else if(*state < xcsf->MIN_CON) {
		*state = xcsf->MIN_CON;
	}
}
