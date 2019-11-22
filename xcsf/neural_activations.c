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
 * @file neural_activations.c
 * @brief Neural network activation functions.
 */ 
 
#include <stdlib.h>
#include <stdio.h>
#include "neural_activations.h"
 
double neural_activate(int function, double state)
{
    switch(function) {
        case LOGISTIC: return logistic_activate(state);
        case RELU: return relu_activate(state);
        case GAUSSIAN: return gaussian_activate(state);
        case TANH: return tanh_activate(state);
        case SIN: return sin_activate(state);
        case COS: return cos_activate(state);
        case SOFT_PLUS: return soft_plus_activate(state);
        case IDENTITY: return identity_activate(state);
        case LEAKY: return leaky_activate(state);
        case SELU: return selu_activate(state);
        default:
            printf("neural_activate(): invalid activation function: %d\n", function);
            exit(EXIT_FAILURE);
    }
}
  
double neural_gradient(int function, double state)
{
    switch(function) {
        case LOGISTIC: return logistic_gradient(state);
        case RELU: return relu_gradient(state);
        case GAUSSIAN: return gaussian_gradient(state);
        case TANH: return tanh_gradient(state);
        case SIN: return sin_gradient(state);
        case COS: return cos_gradient(state);
        case SOFT_PLUS: return soft_plus_gradient(state);
        case IDENTITY: return identity_gradient(state);
        case LEAKY: return leaky_gradient(state);
        case SELU: return selu_gradient(state);
        default:
            printf("neural_gradient(): invalid activation function: %d\n", function);
            exit(EXIT_FAILURE);
    }
}

char *activation_string(int function)
{
     switch(function) {
        case LOGISTIC: return "logistic";
        case RELU: return "relu";
        case GAUSSIAN: return "gaussian";
        case TANH: return "tanh"; 
        case SIN: return "sin";
        case COS: return "cos";
        case SOFT_PLUS: return "soft_plus";
        case IDENTITY: return "identity";
        case LEAKY: return "leaky";
        case SELU: return "selu";
        default:
            printf("activation_string(): invalid activation function: %d\n", function);
            exit(EXIT_FAILURE);
    }
}
