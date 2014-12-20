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

#ifdef NEURAL_PREDICTION
 
typedef struct PRED_NEURON {
	double output;
	double state;
	double *weights;
	double *weights_change;
	double *input;
	int num_inputs;
} PRED_NEURON;

typedef struct PRED {
	int num_layers; // input layer + number of hidden layers + output layer
	int *num_neurons; // number of neurons in each layer
	PRED_NEURON **layer; // neural network
	double pre;
} PRED;

#endif
