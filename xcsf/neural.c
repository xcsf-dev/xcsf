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
#include "data_structures.h"
#include "random.h"
#include "neural.h"

#define MAX_LAYERS 10
#define MAX_NEURONS 50

double neuron_propagate(XCSF *xcsf, NEURON *n, double *input);
void neuron_init(XCSF *xcsf, NEURON *n, int num_inputs, double (*aptr)(double));
void neuron_learn(XCSF *xcsf, NEURON *n, double error);

void neural_init(XCSF *xcsf, BPN *bpn, int layers, int *neurons, double (**aptr)(double))
{
	// set number of layers
	bpn->num_layers = layers;
	// set number of neurons in each layer
	bpn->num_neurons = malloc(sizeof(int)*bpn->num_layers);
	memcpy(bpn->num_neurons, neurons, sizeof(int)*bpn->num_layers);
	// array offsets by 1 since input layer is assumed   
	// malloc layers
	bpn->layer = (NEURON**) malloc((bpn->num_layers-1)*sizeof(NEURON*));
	for(int l = 1; l < bpn->num_layers; l++) {
		bpn->layer[l-1] = (NEURON*) malloc(bpn->num_neurons[l]*sizeof(NEURON));
	}
	// malloc neurons in each other layer
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			neuron_init(xcsf, &bpn->layer[l-1][i], bpn->num_neurons[l-1], aptr[l-1]);
		}
	}   
}

void neural_rand(XCSF *xcsf, BPN *bpn)
{
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			NEURON *n = &bpn->layer[l-1][i];
			for(int w = 0; w < n->num_inputs+1; w++) {
				n->weights[w] = (drand()*2.0)-1.0;
			}
		}
	}    
	(void)xcsf;
}

void neural_propagate(XCSF *xcsf, BPN *bpn, double *input)
{
	double tmpOut[MAX_LAYERS][MAX_NEURONS];
	memcpy(tmpOut[0], input, bpn->num_neurons[0]*sizeof(double));
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			tmpOut[l][i] = neuron_propagate(xcsf, &bpn->layer[l-1][i], tmpOut[l-1]);
		}
	}
}

double neural_output(XCSF *xcsf, BPN *bpn, int i)
{
	(void)xcsf;
	return bpn->layer[bpn->num_layers-2][i].output;
}

void neural_learn(XCSF *xcsf, BPN *bpn, double *output, double *state)
{
	// network already propagated state in set_pred()
	// neural_propagate(bpn, state);
	(void)state; // remove unused parameter warning

	// output layer
	double out_error[bpn->num_neurons[bpn->num_layers-1]];
	int o = bpn->num_layers-2;
	for(int i = 0; i < bpn->num_neurons[bpn->num_layers-1]; i++) {
		out_error[i] = (output[i] - bpn->layer[o][i].output);
		neuron_learn(xcsf, &bpn->layer[o][i], out_error[i]);
	}
	// hidden layers
	double *prev_error = out_error;
	for(int l = bpn->num_layers-2; l > 0; l--) {
		double error[bpn->num_neurons[l]];
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			error[i] = 0.0;
		}
		for(int j = 0; j < bpn->num_neurons[l]; j++) {
			// this neuron's error uses the next layer's error
			for(int k = 0; k < bpn->num_neurons[l+1]; k++) {
				error[j] += prev_error[k] * bpn->layer[l][k].weights[j];
			}
			neuron_learn(xcsf, &bpn->layer[l-1][j], error[j]);
		}
		prev_error = error;
	}    
}

void neural_free(XCSF *xcsf, BPN *bpn)
{
	(void)xcsf;
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
	for(int l = 1; l < bpn->num_layers; l++) {
		free(bpn->layer[l-1]);
	}
	// free pointers to layers
	free(bpn->layer);
	free(bpn->num_neurons);    
}

void neural_print(XCSF *xcsf, BPN *bpn)
{
	printf("neural weights:");
	for(int l = 1; l < bpn->num_layers; l++) {
		for(int i = 0; i < bpn->num_neurons[l]; i++) {
			NEURON *n = &bpn->layer[l-1][i];
			for(int w = 0; w < n->num_inputs+1; w++) {
				printf(" %5f, ", n->weights[w]);
			}
		}
	}
	printf("\n");       
	(void)xcsf;
}

void neural_copy(XCSF *xcsf, BPN *to, BPN *from)
{                                  	
	to->num_layers = from->num_layers;
	memcpy(to->num_neurons, from->num_neurons, sizeof(int)*from->num_layers);
	for(int l = 1; l < from->num_layers; l++) {
		for(int i = 0; i < from->num_neurons[l]; i++) {
			NEURON *a = &to->layer[l-1][i];
			NEURON *b = &from->layer[l-1][i];
			a->activation_ptr = b->activation_ptr;
			a->output = b->output;
			a->state = b->state;
			memcpy(a->weights, b->weights, sizeof(double)*b->num_inputs+1);
			memcpy(a->weights_change, b->weights_change, sizeof(double)*b->num_inputs+1);
			memcpy(a->input, b->input, sizeof(double)*b->num_inputs);
			a->num_inputs = b->num_inputs;
		}
	}    
	(void)xcsf;
}

void neuron_init(XCSF *xcsf, NEURON *n, int num_inputs, double (*aptr)(double))
{
	n->activation_ptr = aptr;
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
	(void)xcsf;
}

double neuron_propagate(XCSF *xcsf, NEURON *n, double *input)
{
	(void)xcsf;
	n->state = 0.0;
	for(int i = 0; i < n->num_inputs; i++) {
		n->input[i] = input[i];
		n->state += n->weights[i] * input[i];
	}
	n->state += n->weights[n->num_inputs];
	n->output = (n->activation_ptr)(n->state);
	return n->output;
}

void neuron_learn(XCSF *xcsf, NEURON *n, double error)
{
	int i;
	for(i = 0; i < n->num_inputs; i++) {
		n->weights_change[i] = error * n->input[i] * xcsf->BETA;
		n->weights[i] += n->weights_change[i];
	}
	n->weights_change[i] = error * xcsf->BETA;
	n->weights[i] += n->weights_change[i];
}  

double sig(double x)
{
	// bipolar logistic function: outputs [-1,1]
	return 2.0 / (1.0 + exp(-x)) - 1.0;
}

double d1sig(double x)
{
	double r = exp(-x);
	return (2.0 * r) / ((r + 1.0) * (r + 1.0));
}

double sig_plain(double x)
{
	// plain sigmoid: outputs [0,1]
	return 1.0 / (1.0 + exp(-x));
}

double gaussian(double x)
{
	return exp((-x * x) / 2.0);
} 

double relu(double x)
{
	return fmax(0.0, x);
}

double bent_identity(double x)
{
	return ((sqrt(x*x+1.0)-1.0)/2.0)+x;
}
