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
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "random.h"
#include "bpn.h"
#include "cons.h"

typedef struct NEURON {
	double output;
	double state;
	double *weights;
	double *weights_change;
	double *input;
	int num_inputs;
} NEURON;

void neuron_init(NEURON *n, int num_inputs);
double propagate_neuron(NEURON *n, double *input);

int num_layers; // input layer + number of hidden layers + output layer
int *num_neurons; // number of neurons in each layer
NEURON **layer; // neural network

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
	n->weights_change = malloc((num_inputs+1)*sizeof(double));
	n->input = malloc(num_inputs*sizeof(double));
	// randomise weights [-0.1,0.1]
	for(int w = 0; w < num_inputs+1; w++) {
		n->weights[w] = 0.2 * (drand() - 0.5);
		n->weights_change[w] = 0.0;
	}
}

void neural_propagate(double *input)
{
	double *output[num_layers];
	for(int l = 0; l < num_layers; l++)
		output[l] = malloc(num_neurons[l]*sizeof(double));
	memcpy(output[0], input, num_neurons[0]*sizeof(double));
	for(int l = 1; l < num_layers; l++) {
		for(int i = 0; i < num_neurons[l]; i++) {
			output[l][i] = propagate_neuron(&layer[l-1][i], output[l-1]);
		}
	}
	for(int l = 0; l < num_layers; l++)
		free(output[l]);
}

double neural_output(int i)
{
	return layer[num_layers-2][i].output;
}

double propagate_neuron(NEURON *n, double *input)
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
			free(n->weights_change);
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
