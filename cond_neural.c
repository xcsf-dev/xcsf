/*
 * Copyright (C) 2012--2015 Richard Preen <rpreen@gmail.com>
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
 * The neural classifier condition module.
 *
 * Provides functionality to create MLP neural networks that compute whether
 * the classifier matches for a given problem instance. Includes operations for
 * copying, mutating, printing, etc.
 */

//#ifdef NEURAL_CONDITIONS
#ifndef RECTANGLE_CONDITIONS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

double cond_neural_output(COND *cond, int i);
double cond_neuron_propagate(NEURON *n, double *input);
void cond_neural_propagate(COND *cond, double *input);
void cond_neuron_init(NEURON *n, int num_inputs);

void cond_init(COND *cond)
{
	// set number of layers
	cond->num_layers = 3;
	// set number of neurons in each layer
	int neurons[3] = {state_length, NUM_HIDDEN_NEURONS, 1};
	cond->num_neurons = malloc(sizeof(int)*cond->num_layers);
	memcpy(cond->num_neurons, neurons, sizeof(int)*cond->num_layers);
	// array offsets by 1 since input layer is assumed   
	// malloc layers
	cond->layer = (NEURON**) malloc((cond->num_layers-1)*sizeof(NEURON*));
	for(int l = 1; l < cond->num_layers; l++) 
		cond->layer[l-1] = (NEURON*) malloc(cond->num_neurons[l]*sizeof(NEURON));
	// malloc neurons in each layer
	for(int l = 1; l < cond->num_layers; l++) {
		for(int i = 0; i < cond->num_neurons[l]; i++)
			cond_neuron_init(&cond->layer[l-1][i], cond->num_neurons[l-1]);
	}
#ifdef SELF_ADAPT_MUTATION
	sam_init(&cond->mu);
#endif
}

void cond_free(COND *cond)
{
	// free neurons
	for(int l = 1; l < cond->num_layers; l++) {
		for(int i = 0; i < cond->num_neurons[l]; i++) {
			NEURON *n = &cond->layer[l-1][i];
			free(n->weights);
			free(n->input);
		}
	}
	// free layers
	for(int l = 1; l < cond->num_layers; l++) 
		free(cond->layer[l-1]);
	// free pointers to layers
	free(cond->layer);
	free(cond->num_neurons);    
#ifdef SELF_ADAPT_MUTATION
	sam_free(cond->mu);
#endif
}

void cond_copy(COND *to, COND *from)
{
	to->num_layers = from->num_layers;
	memcpy(to->num_neurons, from->num_neurons, sizeof(int)*from->num_layers);
	for(int l = 1; l < from->num_layers; l++) {
		for(int i = 0; i < from->num_neurons[l]; i++) {
			NEURON *a = &to->layer[l-1][i];
			NEURON *b = &from->layer[l-1][i];
			a->output = b->output;
			a->state = b->state;
			memcpy(a->weights, b->weights, sizeof(double)*a->num_inputs+1);
			memcpy(a->input, b->input, sizeof(double)*a->num_inputs);
			a->num_inputs = b->num_inputs;
		}
	}
#ifdef SELF_ADAPT_MUTATION
	memcpy(to->mu, from->mu, sizeof(double)*NUM_MU);
#endif
}

void cond_rand(COND *cond)
{
	for(int l = 1; l < cond->num_layers; l++) {
		for(int i = 0; i < cond->num_neurons[l]; i++) {
			NEURON *n = &cond->layer[l-1][i];
			for(int w = 0; w < n->num_inputs; w++) 
				n->weights[w] = (drand()*2.0)-1.0;
		}
	}
}

void cond_cover(COND *cond, double *state)
{
	// generates random weights until the network matches for input state
	do {
		cond_rand(cond);
	} while(!cond_match(cond, state));
}

_Bool cond_match(COND *cond, double *state)
{
	// classifier matches if the first output neuron > 0.5
	cond_neural_propagate(cond, state);
	if(cond_neural_output(cond, 0) > 0.5)
		return true;
	return false;
}

_Bool cond_mutate(COND *cond)
{
	_Bool mod = false;
	double step = S_MUTATION;
#ifdef SELF_ADAPT_MUTATION
	sam_adapt(cond->mu);
	if(NUM_MU > 0) {
		P_MUTATION = cond->mu[0];
		if(NUM_MU > 1)
			step = cond->mu[1];
	}
#endif
	for(int l = 1; l < cond->num_layers; l++) {
		for(int i = 0; i < cond->num_neurons[l]; i++) {
			NEURON *n = &cond->layer[l-1][i];
			for(int w = 0; w < n->num_inputs; w++) {
				if(drand() < P_MUTATION) {
					n->weights[w] = ((drand()*2.0)-1.0)*step;
					if(n->weights[w] > 1.0)
						n->weights[w] = 1.0;
					else if(n->weights[w] < -1.0)
						n->weights[w] = -1.0;
					mod = true;
				}
			}
		}
	}
	return mod;
}

_Bool cond_crossover(COND *cond1, COND *cond2)
{
	// remove unused parameter warnings
	(void)cond1;
	(void)cond2;
	return false;
}

_Bool cond_subsumes(COND *cond1, COND *cond2)
{
	// remove unused parameter warnings
	(void)cond1;
	(void)cond2;
	return false;
}

_Bool cond_general(COND *cond1, COND *cond2)
{
	// remove unused parameter warnings
	(void)cond1;
	(void)cond2;
	return false;
}   

void cond_print(COND *cond)
{
	printf("neural weights:");
	for(int l = 1; l < cond->num_layers; l++) {
		for(int i = 0; i < cond->num_neurons[l]; i++) {
			NEURON *n = &cond->layer[l-1][i];
			for(int w = 0; w < n->num_inputs; w++) 
				printf(" %5f, ", n->weights[w]);
		}
	}
	printf("\n");
}  

void cond_neural_propagate(COND *cond, double *input)
{
	double *output[cond->num_layers];
	for(int l = 0; l < cond->num_layers; l++)
		output[l] = malloc(cond->num_neurons[l]*sizeof(double));
	memcpy(output[0], input, cond->num_neurons[0]*sizeof(double));
	for(int l = 1; l < cond->num_layers; l++) {
		for(int i = 0; i < cond->num_neurons[l]; i++) {
			output[l][i] = cond_neuron_propagate(&cond->layer[l-1][i], output[l-1]);
		}
	}
	for(int l = 0; l < cond->num_layers; l++)
		free(output[l]);
}

double cond_neural_output(COND *cond, int i)
{
	return cond->layer[cond->num_layers-2][i].output;
}

double cond_neuron_propagate(NEURON *n, double *input)
{
	n->state = 0.0;
	for(int i = 0; i < n->num_inputs; i++) {
		n->input[i] = input[i];
		n->state += n->weights[i] * input[i];
	}
	n->state += n->weights[n->num_inputs];
	n->output = tanh(n->state);
	return n->output;
}
 
void cond_neuron_init(NEURON *n, int num_inputs)
{
	n->output = 0.0;
	n->state = 0.0;
	n->num_inputs = num_inputs; 
	n->weights = malloc((num_inputs+1)*sizeof(double));
	n->input = malloc(num_inputs*sizeof(double));
}
#endif
