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
 * The MLP neural network with backpropagation training module.
 *
 * Creates a weight vector representing an MLP neural network to calculate the
 * expected value given a problem instance and provides functions to adapt the
 * weights using the backpropagation algorithm.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "neural.h"
 
#define NUM_OUTPUT 1 // only one output
#define NEURAL_THETA 0.2
#define MAX_LAYERS 3
#define MAX_NEURONS 50
 
double neuron_propagate(NEURON *n, double *input);
void neuron_init(NEURON *n, int num_inputs);
void neuron_learn(NEURON *n, double error);
double d1sig(double x);
double sig(double x);

void neural_init(BPN *bpn)
{
 	// set number of layers
	bpn->num_layers = 3;
	// set number of neurons in each layer
	int neurons[3] = {state_length, NUM_HIDDEN_NEURONS, NUM_OUTPUT};
	bpn->num_neurons = malloc(sizeof(int)*bpn->num_layers);
	memcpy(bpn->num_neurons, neurons, sizeof(int)*bpn->num_layers);
	// array offsets by 1 since input layer is assumed   
	// malloc layers
	bpn->layer = (NEURON**) malloc((bpn->num_layers-1)*sizeof(NEURON*));
	for(int l = 1; l < bpn->num_layers; l++) 
		bpn->layer[l-1] = (NEURON*) malloc(bpn->num_neurons[l]*sizeof(NEURON));
	// malloc neurons in each layer
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++)
			neuron_init(&bpn->layer[l-1][i], bpn->num_neurons[l-1]);
	}   
}

void neural_rand(BPN *bpn)
{
 	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			NEURON *n = &bpn->layer[l-1][i];
			for(int w = 0; w < n->num_inputs; w++) 
				n->weights[w] = (drand()*2.0)-1.0;
		}
	}    
}

void neural_propagate(BPN *bpn, double *input)
{
	double tmpOut[MAX_LAYERS][MAX_NEURONS];
	memcpy(tmpOut[0], input, bpn->num_neurons[0]*sizeof(double));
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			tmpOut[l][i] = neuron_propagate(&bpn->layer[l-1][i], tmpOut[l-1]);
		}
	}
}
         
double neural_output(BPN *bpn, int i)
{
	return bpn->layer[bpn->num_layers-2][i].output;
}

void neural_learn(BPN *bpn, double *output, double *state)
{
	// network already propagated state in set_pred()
	// neural_propagate(bpn, state);
	(void)state; // remove unused parameter warning
	
	// output layer
	double out_error[bpn->num_neurons[bpn->num_layers-1]];
	int o = bpn->num_layers-2;
	for(int i = 0; i < bpn->num_neurons[bpn->num_layers-1]; i++) {
		out_error[i] = (output[i] - bpn->layer[o][i].output) * d1sig(bpn->layer[o][i].state);
		neuron_learn(&bpn->layer[o][i], out_error[i]);
	}
	// hidden layers
	double *prev_error = out_error;
	for(int l = bpn->num_layers-2; l > 0; l--) {
		double error[bpn->num_neurons[l]];
		for(int i = 0; i < bpn->num_neurons[l]; i++)
			error[i] = 0.0;
		for(int j = 0; j < bpn->num_neurons[l]; j++) {
			// this neuron's error uses the next layer's error
			for(int k = 0; k < bpn->num_neurons[l+1]; k++)
				error[j] += prev_error[k] * bpn->layer[l][k].weights[j];
			error[j] *= d1sig(bpn->layer[l-1][j].state);
			neuron_learn(&bpn->layer[l-1][j], error[j]);
		}
		prev_error = error;
	}    
}

void neural_free(BPN *bpn)
{
	// free neurons
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			NEURON *n = &bpn->layer[l-1][i];
			free(n->weights);
			free(n->weights_change);
			free(n->input);
		}
	}
	// free layers
	for(int l = 1; l < bpn->num_layers; l++) 
		free(bpn->layer[l-1]);
	// free pointers to layers
	free(bpn->layer);
	free(bpn->num_neurons);    
}

void neural_print(BPN *bpn)
{
 	printf("neural weights:");
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			NEURON *n = &bpn->layer[l-1][i];
			for(int w = 0; w < n->num_inputs; w++) 
				printf(" %5f, ", n->weights[w]);
		}
	}
	printf("\n");       
}

void neural_copy(BPN *to, BPN *from)
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
			memcpy(a->weights_change, b->weights_change, sizeof(double)*a->num_inputs+1);
			memcpy(a->input, b->input, sizeof(double)*a->num_inputs);
			a->num_inputs = b->num_inputs;
		}
	}    
}
 
void neuron_init(NEURON *n, int num_inputs)
{
	n->output = 0.0;
	n->state = 0.0;
	n->num_inputs = num_inputs; 
	n->weights = malloc((num_inputs+1)*sizeof(double));
	n->weights_change = malloc((num_inputs+1)*sizeof(double));
	n->input = malloc(num_inputs*sizeof(double));
	// randomise weights [-0.1,0.1]
	for(int w = 0; w < num_inputs+1; w++) {
		n->weights[w] = 0.2 * (drand() - 0.5);
		n->weights_change[w] = 0.0;
	}
}
 
double neuron_propagate(NEURON *n, double *input)
{
	n->state = 0.0;
	for(int i = 0; i < n->num_inputs; i++) {
		n->input[i] = input[i];
		n->state += n->weights[i] * input[i];
	}
	n->state += n->weights[n->num_inputs];
	n->output = sig(n->state);
	return n->output;
}
 
void neuron_learn(NEURON *n, double error)
{
	int i;
	for(i = 0; i < n->num_inputs; i++) {
		n->weights_change[i] = error * n->input[i] * BETA;
		n->weights[i] += n->weights_change[i];
	}
	n->weights_change[i] = error * BETA;
	n->weights[i] += n->weights_change[i];
}  

double sig(double x)
{
	return 2.0 / (1.0 + exp(-1.0 * x + NEURAL_THETA)) - 1.0;
}

double d1sig(double x)
{
	double r = exp(-1.0 * x + NEURAL_THETA);
	return (2.0 * r) / ((r + 1.0) * (r + 1.0));
}
