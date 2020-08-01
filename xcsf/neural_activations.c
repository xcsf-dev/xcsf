/*
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
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2012--2020.
 * @brief Neural network activation functions.
 */

#include <stdlib.h>
#include <stdio.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_layer.h"
#include "neural_activations.h"

/**
 * @brief Returns the result from applying a specified activation function.
 * @param a The activation function to apply.
 * @param x The input to the activation function.
 * @return The result from applying the activation function.
 */
double neural_activate(int a, double x)
{
    switch(a) {
        case LOGISTIC:
            return logistic_activate(x);
        case RELU:
            return relu_activate(x);
        case GAUSSIAN:
            return gaussian_activate(x);
        case TANH:
            return tanh_activate(x);
        case SIN:
            return sin_activate(x);
        case COS:
            return cos_activate(x);
        case SOFT_PLUS:
            return soft_plus_activate(x);
        case LINEAR:
            return linear_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case SELU:
            return selu_activate(x);
        case LOGGY:
            return loggy_activate(x);
        default:
            printf("neural_activate(): invalid activation function: %d\n", a);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the derivative from applying a specified activation function.
 * @param a The activation function applied.
 * @param x The input to the activation function.
 * @return The derivative from applying the activation function.
 */
double neural_gradient(int a, double x)
{
    switch(a) {
        case LOGISTIC:
            return logistic_gradient(x);
        case RELU:
            return relu_gradient(x);
        case GAUSSIAN:
            return gaussian_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case SIN:
            return sin_gradient(x);
        case COS:
            return cos_gradient(x);
        case SOFT_PLUS:
            return soft_plus_gradient(x);
        case LINEAR:
            return linear_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case SELU:
            return selu_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        default:
            printf("neural_gradient(): invalid activation function: %d\n", a);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the name of a specified activation function.
 * @param a The activation function.
 * @return The name of the activation function.
 */
const char *neural_activation_string(int a)
{
    switch(a) {
        case LOGISTIC:
            return "logistic";
        case RELU:
            return "relu";
        case GAUSSIAN:
            return "gaussian";
        case TANH:
            return "tanh";
        case SIN:
            return "sin";
        case COS:
            return "cos";
        case SOFT_PLUS:
            return "soft_plus";
        case LINEAR:
            return "linear";
        case LEAKY:
            return "leaky";
        case SELU:
            return "selu";
        case LOGGY:
            return "loggy";
        default:
            printf("neural_activation_string(): invalid activation function: %d\n", a);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Applies a specified activation function to an array.
 * @param state The neuron states.
 * @param output The neuron outputs (set by this function).
 * @param n The length of the input array.
 * @param a The activation function.
 */
void neural_activate_array(double *state, double *output, int n, int a)
{
    for(int i = 0; i < n; ++i) {
        state[i] = clamp(NEURON_MIN, NEURON_MAX, state[i]);
        output[i] = neural_activate(a, state[i]);
    }
}

/**
 * @brief Applies a specified gradient function to an array.
 * @param state The neuron states.
 * @param delta The neuron gradients (set by this function).
 * @param n The length of the input array.
 * @param a The activation function.
 */
void neural_gradient_array(const double *state, double *delta, int n, int a)
{
    for(int i = 0; i < n; ++i) {
        delta[i] *= neural_gradient(a, state[i]);
    }
}
