/*
 * Copyright (C) 2012--2019 Richard Preen <rpreen@gmail.com>
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
	double (*activ_ptr)(double);
	double (*deriv_ptr)(double);
} NEURON;

typedef struct BPN {
	int num_layers; // input layer + number of hidden layers + output layer
	int *num_neurons; // number of neurons in each layer
	NEURON **layer; // neural network
} BPN;

double neural_output(XCSF *xcsf, BPN *bpn, int i);
void neural_copy(XCSF *xcsf, BPN *to, BPN *from);
void neural_free(XCSF *xcsf, BPN *bpn);
void neural_learn(XCSF *xcsf, BPN *bpn, double *output, double *state);
void neural_print(XCSF *xcsf, BPN *bpn);
void neural_propagate(XCSF *xcsf, BPN *bpn, double *input);
void neural_rand(XCSF *xcsf, BPN *bpn);
void neural_init(XCSF *xcsf, BPN *bpn, int layers, int *neurons, 
		double (**aptr)(double), double (**dptr)(double));

// activation functions
double logistic(double x);
double logistic_plain(double x);
double gaussian(double x);
double relu(double x);
double bent_identity(double x);
double identity(double x);
double soft_plus(double x);
double d1logistic(double x);
double d1logistic_plain(double x);
double d1tanh(double x);
double d1bent_identity(double x);
double d1gaussian(double x);
double d1relu(double x);
