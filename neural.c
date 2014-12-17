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

#ifdef NEURAL_CONDITIONS
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"
#include "neural.h"

#define NUM_OUTPUT 1 // only one output required for matching

typedef struct NEURON {
	double output;
	double state;
	double *weights;
	double *input;
	int num_inputs;
} NEURON;

double neural_output(int i);
double neuron_propagate(NEURON *n, double *input);
void neural_propagate(double *input);
void neural_set_weights(double *nw);
void neuron_init(NEURON *n, int num_inputs);

int num_layers; // input layer + number of hidden layers + output layer
int *num_neurons; // number of neurons in each layer
NEURON **layer; // neural network
 
void cond_init(CL *c)
{
	c->cond_length = ((state_length+1)*NUM_HIDDEN_NEURONS)
		+((NUM_HIDDEN_NEURONS+1)*NUM_OUTPUT);
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
	for(int i = 0; i < c->cond_length; i++)
		c->cond[i] = (drand()*2.0)-1.0;
}

void cond_cover(CL *c, double *state)
{
	// generates random weights until the network matches for input state
	do {
		for(int i = 0; i < c->cond_length; i++)
			c->cond[i] = (drand()*2.0)-1.0;
	} while(!cond_match(c, state));
}

_Bool cond_match(CL *c, double *state)
{
	// classifier matches if the first output neuron > 0.5
	neural_set_weights(c->cond);
	neural_propagate(state);
	if(neural_output(0) > 0.5)
		return true;
	return false;
}

_Bool cond_mutate(CL *c)
{
	double mod = false;
	double step = S_MUTATION;
#ifdef SELF_ADAPT_MUTATION
	sam_adapt(c);
	if(NUM_MU > 0) {
		P_MUTATION = c->mu[0];
		if(NUM_MU > 1)
			step = c->mu[1];
	}
#endif
	for(int i = 0; i < c->cond_length; i++) {
		if(drand() < P_MUTATION) {
			c->cond[i] += ((drand()*2.0)-1.0)*step;
			if(c->cond[i] > 1.0)
				c->cond[i] = 1.0;
			else if(c->cond[i] < -1.0)
				c->cond[i] = -1.0;
			mod = true;
		}
	}
	return mod;
}

_Bool cond_crossover(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_subsumes(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}

_Bool cond_general(CL *c1, CL *c2)
{
	// remove unused parameter warnings
	(void)c1;
	(void)c2;
	return false;
}   

void cond_print(CL *c)
{
	printf("neural weights:");
	for(int i = 0; i < c->cond_length; i++)
		printf(" %5f, ", c->cond[i]);
	printf("\n");
}  

void neural_init(int layers, int *neurons)
{
	num_layers = layers;
	// set number of neurons in each layer
	num_neurons = malloc(sizeof(int)*num_layers);
	memcpy(num_neurons, neurons, sizeof(int)*num_layers);
	// array offsets by 1 since input layer is assumed   
	layer = (NEURON**) malloc((num_layers-1)*sizeof(NEURON*));
	for(int l = 1; l < num_layers; l++) 
		layer[l-1] = (NEURON*) malloc(num_neurons[l]*sizeof(NEURON));
	for(int l = 1; l < num_layers; l++) {
		for(int i = 0; i < num_neurons[l]; i++)
			neuron_init(&layer[l-1][i], num_neurons[l-1]);
	}
}

void neuron_init(NEURON *n, int num_inputs)
{
	n->output = 0.0;
	n->state = 0.0;
	n->num_inputs = num_inputs; 
	n->weights = malloc((num_inputs+1)*sizeof(double));
	n->input = malloc(num_inputs*sizeof(double));
}

void neural_propagate(double *input)
{
	double *output[num_layers];
	for(int l = 0; l < num_layers; l++)
		output[l] = malloc(num_neurons[l]*sizeof(double));
	memcpy(output[0], input, num_neurons[0]*sizeof(double));
	for(int l = 1; l < num_layers; l++) {
		for(int i = 0; i < num_neurons[l]; i++) {
			output[l][i] = neuron_propagate(&layer[l-1][i], output[l-1]);
		}
	}
	for(int l = 0; l < num_layers; l++)
		free(output[l]);
}

double neural_output(int i)
{
	return layer[num_layers-2][i].output;
}

double neuron_propagate(NEURON *n, double *input)
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

void neural_set_weights(double *nw)
{
	int cnt = 0;
	for(int l = 1; l < num_layers; l++) {
		for(int i = 0; i < num_neurons[l]; i++) {
			for(int w = 0; w < num_neurons[l-1]+1; w++) {
				NEURON *n = &layer[l-1][i];   	
				n->weights[w] = nw[cnt];
				cnt++;
			}
		}
	}
}

void neural_free()
{
	// free neurons
	for(int l = 1; l < num_layers; l++) {
		for(int i = 0; i < num_neurons[l]; i++) {
			NEURON *n = &layer[l-1][i];
			free(n->weights);
			free(n->input);
		}
	}
	// free layers
	for(int l = 1; l < num_layers; l++) 
		free(layer[l-1]);
	// free pointers to layers
	free(layer);
	free(num_neurons);
}    
#endif
