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

#define LOGISTIC 0 
#define RELU 1 
#define TANH 2 
#define IDENTITY 3 
#define GAUSSIAN 4 
#define SIN 5 
#define COS 6
#define SOFT_PLUS 7 
#define BENT_IDENTITY 8 
#define HARDTAN 9
#define STAIR 10
#define LEAKY 11
#define ELU 12
#define RAMP 13
#define NUM_ACTIVATIONS 14

typedef double (*activ_ptr)(double);
typedef double (*deriv_ptr)(double);

typedef struct NEURON {
    double output;
    double state;
    double *weights;
    double *v;
    double *input;
    int num_inputs;
    activ_ptr activ;
    deriv_ptr deriv;
} NEURON;

typedef struct BPN {
    int num_layers; // input layer + number of hidden layers + output layer
    int *num_neurons; // number of neurons in each layer
    NEURON **layer; // neural network
    double **tmp; // temporary storage
} BPN;

double neural_output(XCSF *xcsf, BPN *bpn, int i);
void neural_copy(XCSF *xcsf, BPN *to, BPN *from);
void neural_free(XCSF *xcsf, BPN *bpn);
void neural_learn(XCSF *xcsf, BPN *bpn, double *output, double *state);
void neural_print(XCSF *xcsf, BPN *bpn);
void neural_propagate(XCSF *xcsf, BPN *bpn, double *input);
void neural_rand(XCSF *xcsf, BPN *bpn);
void neural_init(XCSF *xcsf, BPN *bpn, int layers, int *neurons, int *activ);
void neuron_set_activation(XCSF *xcsf, NEURON *n, int func);  
_Bool neural_mutate(XCSF *xcsf, BPN *bpn);

static inline double logistic_activ(double x) {return 1./(1.+exp(-x));}
static inline double logistic_deriv(double x) {return (1-x)*x;}
//static inline double logistic_activ(double x) {return 2./(1+exp(-x))-1;} // bipolar
//static inline double logistic_deriv(double x) {double r=exp(-x); return (2*r)/((r+1)*(r+1));}
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
static inline double cos_deriv(double x) {return -sin(x);}
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
