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
 * The MLP neural network classifier computed prediction module.
 *
 * Creates a weight vector representing an MLP neural network to calculate the
 * expected value given a problem instance and adapts the weights using the
 * backpropagation algorithm.
 */

#ifdef NEURAL_PREDICTION

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "random.h"
#include "cons.h"
#include "cl.h"

#define NUM_OUTPUT 1 // only one output
#define NEURAL_BETA 0.2
#define NEURAL_THETA 0.2

double pred_d1sig(double x);
double pred_neural_output(PRED *pred, int i);
double pred_neuron_propagate(PRED_NEURON *n, double *input);
double pred_sig(double x);
void pred_neuron_init(PRED_NEURON *n, int num_inputs);
void pred_neuron_learn(PRED_NEURON *n, double error);
void pred_neural_propagate(PRED *pred, double *input);

void pred_init(PRED *pred)
{
	// set number of layers
	pred->num_layers = 3;
	// set number of neurons in each layer
	int neurons[3] = {state_length, NUM_HIDDEN_NEURONS, NUM_OUTPUT};
	pred->num_neurons = malloc(sizeof(int)*pred->num_layers);
	memcpy(pred->num_neurons, neurons, sizeof(int)*pred->num_layers);
	// array offsets by 1 since input layer is assumed   
	// malloc layers
	pred->layer = (PRED_NEURON**) malloc((pred->num_layers-1)*sizeof(PRED_NEURON*));
	for(int l = 1; l < pred->num_layers; l++) 
		pred->layer[l-1] = (PRED_NEURON*) malloc(pred->num_neurons[l]*sizeof(PRED_NEURON));
	// malloc neurons in each layer
	for(int l = 1; l < pred->num_layers; l++) {
		for(int i = 0; i < pred->num_neurons[l]; i++)
			pred_neuron_init(&pred->layer[l-1][i], pred->num_neurons[l-1]);
	}
}

void pred_free(PRED *pred)
{
	// free neurons
	for(int l = 1; l < pred->num_layers; l++) {
		for(int i = 0; i < pred->num_neurons[l]; i++) {
			PRED_NEURON *n = &pred->layer[l-1][i];
			free(n->weights);
			free(n->weights_change);
			free(n->input);
		}
	}
	// free layers
	for(int l = 1; l < pred->num_layers; l++) 
		free(pred->layer[l-1]);
	// free pointers to layers
	free(pred->layer);
	free(pred->num_neurons);    
}

void pred_copy(PRED *to, PRED *from)
{
	to->num_layers = from->num_layers;
	memcpy(to->num_neurons, from->num_neurons, sizeof(int)*from->num_layers);
	for(int l = 1; l < from->num_layers; l++) {
		for(int i = 0; i < from->num_neurons[l]; i++) {
			PRED_NEURON *a = &to->layer[l-1][i];
			PRED_NEURON *b = &from->layer[l-1][i];
			a->output = b->output;
			a->state = b->state;
			memcpy(a->weights, b->weights, sizeof(double)*a->num_inputs+1);
			memcpy(a->weights_change, b->weights_change, sizeof(double)*a->num_inputs+1);
			memcpy(a->input, b->input, sizeof(double)*a->num_inputs);
			a->num_inputs = b->num_inputs;
		}
	}
}

void pred_update(PRED *pred, double p, double *state)
{
	double out[1];
	out[0] = p;
	pred_neural_propagate(pred, state);
	// output layer
	double out_error[pred->num_neurons[pred->num_layers-1]];
	int o = pred->num_layers-2;
	for(int i = 0; i < pred->num_neurons[pred->num_layers-1]; i++) {
		out_error[i] = (out[i] - pred->layer[o][i].output) * pred_d1sig(pred->layer[o][i].state);
		pred_neuron_learn(&pred->layer[o][i], out_error[i]);
	}
	// hidden layers
	double *prev_error = out_error;
	for(int l = pred->num_layers-2; l > 0; l--) {
		double error[pred->num_neurons[l]];
		for(int i = 0; i < pred->num_neurons[l]; i++)
			error[i] = 0.0;
		for(int j = 0; j < pred->num_neurons[l]; j++) {
			// this neuron's error uses the next layer's error
			for(int k = 0; k < pred->num_neurons[l+1]; k++)
				error[j] += prev_error[k] * pred->layer[l][k].weights[j];
			error[j] *= pred_d1sig(pred->layer[l-1][j].state);
			pred_neuron_learn(&pred->layer[l-1][j], error[j]);
		}
		prev_error = error;
	}
}

double pred_compute(PRED *pred, double *state)
{
	pred_neural_propagate(pred, state);
	return pred_neural_output(pred, 0);
}


void pred_print(PRED *pred)
{
	printf("neural weights:");
	for(int l = 1; l < pred->num_layers; l++) {
		for(int i = 0; i < pred->num_neurons[l]; i++) {
			PRED_NEURON *n = &pred->layer[l-1][i];
			for(int w = 0; w < n->num_inputs; w++) 
				printf(" %5f, ", n->weights[w]);
		}
	}
	printf("\n");
}  

void pred_neural_propagate(PRED *pred, double *input)
{
	double *output[pred->num_layers];
	for(int l = 0; l < pred->num_layers; l++)
		output[l] = malloc(pred->num_neurons[l]*sizeof(double));
	memcpy(output[0], input, pred->num_neurons[0]*sizeof(double));
	for(int l = 1; l < pred->num_layers; l++) {
		for(int i = 0; i < pred->num_neurons[l]; i++) {
			output[l][i] = pred_neuron_propagate(&pred->layer[l-1][i], output[l-1]);
		}
	}
	for(int l = 0; l < pred->num_layers; l++)
		free(output[l]);
}
         
double pred_neural_output(PRED *pred, int i)
{
	return pred->layer[pred->num_layers-2][i].output;
}
 
void pred_neuron_init(PRED_NEURON *n, int num_inputs)
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
 
double pred_neuron_propagate(PRED_NEURON *n, double *input)
{
	n->state = 0.0;
	for(int i = 0; i < n->num_inputs; i++) {
		n->input[i] = input[i];
		n->state += n->weights[i] * input[i];
	}
	n->state += n->weights[n->num_inputs];
	n->output = pred_sig(n->state);
	return n->output;
}
 
void pred_neuron_learn(PRED_NEURON *n, double error)
{
	int i;
	for(i = 0; i < n->num_inputs; i++) {
		n->weights_change[i] = error * n->input[i] * NEURAL_BETA;
		n->weights[i] += n->weights_change[i];
	}
	n->weights_change[i] = error * NEURAL_BETA;
	n->weights[i] += n->weights_change[i];
}
 
double pred_sig(double x)
{
	return 2.0 / (1.0 + exp(-1.0 * x + NEURAL_THETA)) - 1.0;
}

double pred_d1sig(double x)
{
	double r = exp(-1.0 * x + NEURAL_THETA);
	return (2.0 * r) / ((r + 1.0) * (r + 1.0));
}
 
#endif
