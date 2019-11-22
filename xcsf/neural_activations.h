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
     
/**
 * @file neural_activations.h
 * @brief Neural network activation functions.
 */ 

#pragma once
 
#include <math.h>

#define LOGISTIC 0 
#define RELU 1 
#define TANH 2 
#define IDENTITY 3 
#define GAUSSIAN 4 
#define SIN 5 
#define COS 6
#define SOFT_PLUS 7 
#define LEAKY 8
#define SELU 9
#define NUM_ACTIVATIONS 10
 
double neural_activate(int function, double state);
double neural_gradient(int function, double state);
char *activation_string(int function);

static inline double logistic_activate(double x){return 1./(1.+exp(-x));}
static inline double logistic_gradient(double x){double fx=1./(1.+exp(-x)); return (1-fx)*fx;}
static inline double gaussian_activate(double x){return exp(-x*x);}
static inline double gaussian_gradient(double x){return -2*x*exp(-x*x);}
static inline double relu_activate(double x){return x*(x>0);}
static inline double relu_gradient(double x){return (x>0);}
static inline double selu_activate(double x){return (x>=0)*1.0507*x+(x<0)*1.0507*1.6732*expm1(x);}
static inline double selu_gradient(double x){return (x>=0)*1.0507+(x<0)*(1.6732*exp(x));}
static inline double identity_activate(double x){return x;}
static inline double identity_gradient(double x){(void)x; return 1;}
static inline double soft_plus_activate(double x){return log1p(exp(x));}
static inline double soft_plus_gradient(double x){return 1./(1.+exp(-x));}
static inline double tanh_activate(double x){return tanh(x);}
static inline double tanh_gradient(double x){double t=tanh(x); return 1-t*t;}
static inline double leaky_activate(double x){return (x>0) ? x : .1*x;}
static inline double leaky_gradient(double x){return (x<0) ? .1 : 1;}
static inline double sin_activate(double x){return sin(x);}
static inline double sin_gradient(double x){return cos(x);}
static inline double cos_activate(double x){return cos(x);}
static inline double cos_gradient(double x){return -sin(x);}
