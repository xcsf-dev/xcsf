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
  
typedef struct NEURON {
	double output;
	double state;
	double *weights;
	double *weights_change;
	double *input;
	int num_inputs;
	double (*activation_ptr)(double);
} NEURON;

typedef struct BPN {
    int num_layers; // input layer + number of hidden layers + output layer
	int *num_neurons; // number of neurons in each layer
	NEURON **layer; // neural network
} BPN;
 
double neural_output(BPN *bpn, int i);
void neural_copy(BPN *to, BPN *from);
void neural_free(BPN *bpn);
void neural_learn(BPN *bpn, double *output, double *state);
void neural_print(BPN *bpn);
void neural_propagate(BPN *bpn, double *input);
void neural_rand(BPN *bpn);
void neural_init(BPN *bpn, int layers, int *neurons, double (**aptr)(double));

// activation functions
double d1sig(double x);
double sig(double x);
double sig_plain(double x);
double gaussian(double x);
double relu(double x);
double bent_identity(double x);
