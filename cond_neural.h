/*
 * Copyright (C) 2012--2015 Richard Preen <rpreen@gmail.com>
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

//#ifdef NEURAL_CONDITIONS
#ifndef RECTANGLE_CONDITIONS

typedef struct NEURON {
	double output;
	double state;
	double *weights;
	double *input;
	int num_inputs;
} NEURON;

typedef struct COND {
	int num_layers; // input layer + number of hidden layers + output layer
	int *num_neurons; // number of neurons in each layer
	NEURON **layer; // neural network
#ifdef SELF_ADAPT_MUTATION
	double *mu;
#endif
} COND;

#endif
