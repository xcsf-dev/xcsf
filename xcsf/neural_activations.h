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

#include <math.h>

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
 
typedef double (*activate_ptr)(double);
typedef double (*gradient_ptr)(double);

void activation_set(activate_ptr *activate, int func);
void gradient_set(gradient_ptr *gradient, int func);

static inline double logistic_activate(double x) {return 1./(1.+exp(-x));}
static inline double logistic_gradient(double x) {return (1-x)*x;}
//static inline double logistic_activate(double x) {return 2./(1+exp(-x))-1;} // bipolar
//static inline double logistic_gradient(double x) {double r=exp(-x); return (2*r)/((r+1)*(r+1));}
static inline double gaussian_activate(double x) {return exp(-x*x);}
static inline double gaussian_gradient(double x) {return -2*x*exp((-x*x)/2.);}
static inline double relu_activate(double x) {return x*(x>0);}
static inline double relu_gradient(double x) {return (x > 0);}
static inline double bent_identity_activate(double x) {return ((sqrt(x*x+1)-1)/2.)+x;}
static inline double bent_identity_gradient(double x) {return (2*sqrt(x*x+1)/x)+1;}
static inline double identity_activate(double x) {return x;}
static inline double identity_gradient(double x) {(void)x; return 1;}
static inline double soft_plus_activate(double x) {return log1p(exp(x));}
static inline double tanh_activate(double x) {return (expm1(2*x))/(exp(2*x)+1);}
static inline double tanh_gradient(double x) {return 1-x*x;}
static inline double logistic_plain(double x) {return (1-x)*x;}
static inline double leaky_activate(double x) {return (x>0) ? x : .1*x;}
static inline double leaky_gradient(double x) {return (x>0)+.1;}
static inline double elu_activate(double x) {return (x >= 0)*x + (x < 0)*expm1(x);}
static inline double elu_gradient(double x) {return (x >= 0) + (x < 0)*(x + 1);}
static inline double ramp_activate(double x) {return x*(x>0)+.1*x;}
static inline double ramp_gradient(double x) {return (x>0)+.1;}
static inline double sin_activate(double x) {return sin(x);}
static inline double sin_gradient(double x) {return cos(x);}
static inline double cos_activate(double x) {return cos(x);}
static inline double cos_gradient(double x) {return -sin(x);}
static inline double stair_activate(double x)
{
    int n = floor(x);
    if (n%2 == 0) {return floor(x/2.);}
    else {return (x-n)+floor(x/2.);}
}
static inline double stair_gradient(double x)
{
    if(floor(x) == x) {return 0;}
    return 1;
}
static inline double hardtan_activate(double x)
{
    if (x < -1) {return -1;}
    if (x > 1) {return 1;}
    return x;
}
static inline double hardtan_gradient(double x)
{
    if (x > -1 && x < 1) {return 1;}
    return 0;
}
