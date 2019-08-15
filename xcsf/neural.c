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
#include <float.h>
#include "data_structures.h"
#include "random.h"
#include "neural.h"

double neuron_propagate(XCSF *xcsf, NEURON *n, double *input);
void neuron_init(XCSF *xcsf, NEURON *n, int num_inputs, int func);
void neuron_learn(XCSF *xcsf, NEURON *n, double error);

void neural_init(XCSF *xcsf, BPN *bpn, int layers, int *neurons, int *activ)
{
    // set number of hidden and output layers
    bpn->num_layers = layers;
    // set number of neurons in each layer
    bpn->num_neurons = malloc(sizeof(int)*bpn->num_layers);
    memcpy(bpn->num_neurons, neurons, sizeof(int)*bpn->num_layers);
    // malloc layers
    bpn->layer = (NEURON**) malloc((bpn->num_layers)*sizeof(NEURON*));
    for(int i = 0; i < bpn->num_layers; i++) {
        bpn->layer[i] = (NEURON*) malloc(bpn->num_neurons[i]*sizeof(NEURON));
    }
    // initialise first hidden layer
    for(int i = 0; i < bpn->num_neurons[0]; i++) {
        neuron_init(xcsf, &bpn->layer[0][i], xcsf->num_x_vars, activ[0]);
    }
    // initialise each other layer
    for(int i = 1; i < bpn->num_layers; i++) {
        for(int j = 0; j < bpn->num_neurons[i]; j++) {
            neuron_init(xcsf, &bpn->layer[i][j], bpn->num_neurons[i-1], activ[i]);
        }
    }   
    // temporary storage
    bpn->tmp = malloc(sizeof(double*)*bpn->num_layers);
    for(int i = 0; i < bpn->num_layers; i++) {
        bpn->tmp[i] = malloc(sizeof(double)*bpn->num_neurons[i]);
    }
}

void neural_rand(XCSF *xcsf, BPN *bpn)
{
    for(int i = 0; i < bpn->num_layers; i++) {
        for(int j = 0; j < bpn->num_neurons[i]; j++) {
            NEURON *n = &bpn->layer[i][j];
            for(int k = 0; k < n->num_inputs+1; k++) {
                n->weights[k] = (drand()*2.0)-1.0;
            }
        }
    }    
    (void)xcsf;
}

void neural_propagate(XCSF *xcsf, BPN *bpn, double *input)
{
    // propagate inputs
    for(int i = 0; i < bpn->num_neurons[0]; i++) {
        bpn->tmp[0][i] = neuron_propagate(xcsf, &bpn->layer[0][i], input);
    }
    // propagate hidden and output layers
    for(int i = 1; i < bpn->num_layers; i++) {
        for(int j = 0; j < bpn->num_neurons[i]; j++) {
            bpn->tmp[i][j] = neuron_propagate(xcsf, &bpn->layer[i][j], bpn->tmp[i-1]);
        }
    }
}

double neural_output(XCSF *xcsf, BPN *bpn, int i)
{
    (void)xcsf;
    return bpn->layer[bpn->num_layers-1][i].output;
}

void neural_learn(XCSF *xcsf, BPN *bpn, double *output, double *state)
{
    // network already propagated state in set_pred()
    // neural_propagate(xcsf, bpn, state);
    (void)state; // remove unused parameter warning

    // output layer (errors = deltas)
    double *out_error = bpn->tmp[bpn->num_layers-1];
    for(int i = 0; i < bpn->num_neurons[bpn->num_layers-1]; i++) {
        NEURON * neuro = &bpn->layer[bpn->num_layers-1][i];
        out_error[i] = (output[i] - neuro->output) * (neuro->deriv_ptr)(neuro->state);
        neuron_learn(xcsf, neuro, out_error[i]);
    }
    // hidden layers
    double *prev_error = out_error;
    for(int i = bpn->num_layers-2; i >= 0; i--) {
        double *error = bpn->tmp[i];
        for(int j = 0; j < bpn->num_neurons[i]; j++) {
            // this neuron's error uses the next layer's error
            error[j] = 0.0;
            for(int k = 0; k < bpn->num_neurons[i+1]; k++) {
                error[j] += prev_error[k] * bpn->layer[i+1][k].weights[j];
            }
            NEURON * neuro = &bpn->layer[i][j];
            error[j] *= (neuro->deriv_ptr)(neuro->state);
            neuron_learn(xcsf, neuro, error[j]);
        }
        prev_error = error;
    }
}

void neural_free(XCSF *xcsf, BPN *bpn)
{
    (void)xcsf;
    // free neurons
    for(int i = 0; i < bpn->num_layers; i++) {
        for(int j = 0; j < bpn->num_neurons[i]; j++) {
            NEURON *n = &bpn->layer[i][j];
            free(n->weights);
            free(n->weights_change);
            free(n->input);
        }
    }
    // free layers
    for(int i = 0; i < bpn->num_layers; i++) {
        free(bpn->layer[i]);
    }
    // free pointers to layers
    free(bpn->layer);
    free(bpn->num_neurons);    
    // free temporary storage
    for(int i = 0; i < bpn->num_layers; i++) {
        free(bpn->tmp[i]);
    }
    free(bpn->tmp);
}

void neural_print(XCSF *xcsf, BPN *bpn)
{
    printf("neural weights:");
    for(int i = 0; i < bpn->num_layers; i++) {
        for(int j = 0; j < bpn->num_neurons[i]; j++) {
            NEURON *n = &bpn->layer[i][j];
            for(int k = 0; k < n->num_inputs+1; k++) {
                printf(" %5f, ", n->weights[k]);
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
    for(int i = 0; i < from->num_layers; i++) {
        for(int j = 0; j < from->num_neurons[i]; j++) {
            NEURON *a = &to->layer[i][j];
            NEURON *b = &from->layer[i][j];
            a->activ_ptr = b->activ_ptr;
            a->deriv_ptr = b->deriv_ptr;
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

void neuron_init(XCSF *xcsf, NEURON *n, int num_inputs, int func)
{
    neuron_set_activation(xcsf, n, func);
    n->output = 0.0;
    n->state = 0.0;
    n->num_inputs = num_inputs; 
    n->weights = malloc((num_inputs+1)*sizeof(double));
    n->weights_change = malloc((num_inputs+1)*sizeof(double));
    n->input = malloc(num_inputs*sizeof(double));
    // randomise weights [-0.1,0.1]
    for(int i = 0; i < num_inputs+1; i++) {
        n->weights[i] = 0.2 * (drand() - 0.5);
        n->weights_change[i] = 0.0;
    }
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
    n->output = (n->activ_ptr)(n->state);
    n->output = fmax(xcsf->MIN_CON, fmin(xcsf->MAX_CON, n->output));
    return n->output;
}

void neuron_learn(XCSF *xcsf, NEURON *n, double error)
{
    for(int i = 0; i < n->num_inputs; i++) {
        n->weights[i] += xcsf->MOMENTUM * n->weights_change[i];
        n->weights_change[i] = error * n->input[i] * xcsf->XCSF_ETA;
        n->weights[i] += n->weights_change[i];
    }
    n->weights[n->num_inputs] += xcsf->MOMENTUM * n->weights_change[n->num_inputs];
    n->weights_change[n->num_inputs] = error * xcsf->XCSF_ETA;
    n->weights[n->num_inputs] += n->weights_change[n->num_inputs];
}

// bipolar logistic sigmoid function: outputs [-1,1]
static inline double logistic_activ(double x) {return 2./(1+exp(-x))-1;}
static inline double logistic_deriv(double x) {double y = (x+1.)/2.; return 2*(1-y)*y;}
static inline double gaussian_activ(double x) {return exp(-x*x);}
static inline double gaussian_deriv(double x) {return -2*x*exp((-x*x)/2.);}
static inline double relu_activ(double x) {return x*(x>0);}
static inline double relu_deriv(double x) {return (x > 0);}
static inline double bent_identity_activ(double x) {return ((sqrt(x*x+1)-1)/2.)+x;}
static inline double bent_identity_deriv(double x) {return (2*sqrt(x*x+1)/x)+1;}
static inline double identity_activ(double x) {return x;}
static inline double identity_deriv(double x) {(void)x; return 1;}
static inline double soft_plus_activ(double x) {return log1p(exp(x));}
static inline double tanh_activ(double x) {return (expm1(2*x))/(exp(2*x)+1);}
static inline double tanh_deriv(double x) {return 1-x*x;}
static inline double logistic_plain(double x) {return (1-x)*x;}
static inline double leaky_activ(double x) {return (x>0) ? x : .1*x;}
static inline double leaky_deriv(double x) {return (x>0)+.1;}
static inline double elu_activ(double x) {return (x >= 0)*x + (x < 0)*expm1(x);}
static inline double elu_deriv(double x) {return (x >= 0) + (x < 0)*(x + 1);}
static inline double ramp_activ(double x) {return x*(x>0)+.1*x;}
static inline double ramp_deriv(double x) {return (x>0)+.1;}
static inline double stair_activ(double x)
{
    int n = floor(x);
    if (n%2 == 0) {return floor(x/2.);}
    else {return (x-n)+floor(x/2.);}
}
static inline double stair_deriv(double x)
{
    if(floor(x) == x) {return 0;}
    return 1;
}
static inline double hardtan_activ(double x)
{
    if (x < -1) {return -1;}
    if (x > 1) {return 1;}
    return x;
}
static inline double hardtan_deriv(double x)
{
    if (x > -1 && x < 1) {return 1;}
    return 0;
}
 
void neuron_set_activation(XCSF *xcsf, NEURON *n, int func)
{
    switch(func) {
        case LOGISTIC:
            n->activ_ptr = &logistic_activ;
            n->deriv_ptr = &logistic_deriv;
            break;
        case RELU:
            n->activ_ptr = &relu_activ;
            n->deriv_ptr = &relu_deriv;
            break;
        case GAUSSIAN:
            n->activ_ptr = &gaussian_activ;
            n->deriv_ptr = &gaussian_deriv;
            break;
        case BENT_IDENTITY:
            n->activ_ptr = &bent_identity_activ;
            n->deriv_ptr = &bent_identity_deriv;
            break;
        case TANH:
            n->activ_ptr = &tanh_activ;
            n->deriv_ptr = &tanh_deriv;
            break;
        case SIN:
            n->activ_ptr = &sin;
            n->deriv_ptr = &cos;
            break;
        case SOFT_PLUS:
            n->activ_ptr = &soft_plus_activ;
            n->deriv_ptr = &logistic_plain;
            break;
        case IDENTITY:
            n->activ_ptr = &identity_activ;
            n->deriv_ptr = &identity_deriv;
            break;
        case HARDTAN:
            n->activ_ptr = &hardtan_activ;
            n->deriv_ptr = &hardtan_deriv;
            break;
        case STAIR:
            n->activ_ptr = &stair_activ;
            n->deriv_ptr = &stair_deriv;
            break;
        case LEAKY:
            n->activ_ptr = &leaky_activ;
            n->deriv_ptr = &leaky_deriv;
            break;
        case ELU:
            n->activ_ptr = &elu_activ;
            n->deriv_ptr = &elu_deriv;
            break;
        case RAMP:
            n->activ_ptr = &ramp_activ;
            n->deriv_ptr = &ramp_deriv;
            break;
        default:
            printf("error: invalid activation function: %d\n", func);
            exit(EXIT_FAILURE);
    }                                    
    (void)xcsf;
}
