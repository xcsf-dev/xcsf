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

#include "neural_activations.h"
#include "neural_layer.h"
#include "utils.h"

/**
 * @brief Returns the result from applying a specified activation function.
 * @param [in] a The activation function to apply.
 * @param [in] x The input to the activation function.
 * @return The result from applying the activation function.
 */
double
neural_activate(const int a, const double x)
{
    switch (a) {
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
 * @param [in] a The activation function applied.
 * @param [in] x The input to the activation function.
 * @return The derivative from applying the activation function.
 */
double
neural_gradient(const int a, const double x)
{
    switch (a) {
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
 * @param [in] a The activation function.
 * @return The name of the activation function.
 */
const char *
neural_activation_string(const int a)
{
    switch (a) {
        case LOGISTIC:
            return STRING_LOGISTIC;
        case RELU:
            return STRING_RELU;
        case GAUSSIAN:
            return STRING_GAUSSIAN;
        case TANH:
            return STRING_TANH;
        case SIN:
            return STRING_SIN;
        case COS:
            return STRING_COS;
        case SOFT_PLUS:
            return STRING_SOFT_PLUS;
        case LINEAR:
            return STRING_LINEAR;
        case LEAKY:
            return STRING_LEAKY;
        case SELU:
            return STRING_SELU;
        case LOGGY:
            return STRING_LOGGY;
        case SOFT_MAX:
            return STRING_SOFT_MAX;
        default:
            printf("neural_activation_string(): invalid activation: %d\n", a);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the integer representation of an activation function.
 * @param [in] a String representing the name of an activation function.
 * @return Integer representing an activation function.
 */
int
neural_activation_as_int(const char *a)
{
    if (strncmp(a, STRING_LOGISTIC, 9) == 0) {
        return LOGISTIC;
    }
    if (strncmp(a, STRING_RELU, 5) == 0) {
        return RELU;
    }
    if (strncmp(a, STRING_GAUSSIAN, 9) == 0) {
        return GAUSSIAN;
    }
    if (strncmp(a, STRING_TANH, 5) == 0) {
        return TANH;
    }
    if (strncmp(a, STRING_SIN, 4) == 0) {
        return SIN;
    }
    if (strncmp(a, STRING_COS, 4) == 0) {
        return COS;
    }
    if (strncmp(a, STRING_SOFT_PLUS, 10) == 0) {
        return SOFT_PLUS;
    }
    if (strncmp(a, STRING_LINEAR, 7) == 0) {
        return LINEAR;
    }
    if (strncmp(a, STRING_LEAKY, 6) == 0) {
        return LEAKY;
    }
    if (strncmp(a, STRING_SELU, 5) == 0) {
        return SELU;
    }
    if (strncmp(a, STRING_LOGGY, 6) == 0) {
        return LOGGY;
    }
    if (strncmp(a, STRING_SOFT_MAX, 9) == 0) {
        return SOFT_MAX;
    }
    printf("neural_activation_as_int(): invalid activation: %s\n", a);
    exit(EXIT_FAILURE);
}

/**
 * @brief Applies an activation function to a vector of neuron states.
 * @param [in,out] state The neuron states.
 * @param [in,out] output The neuron outputs.
 * @param [in] n The length of the input array.
 * @param [in] a The activation function.
 */
void
neural_activate_array(double *state, double *output, const int n, const int a)
{
    for (int i = 0; i < n; ++i) {
        state[i] = clamp(state[i], NEURON_MIN, NEURON_MAX);
        output[i] = neural_activate(a, state[i]);
    }
}

/**
 * @brief Applies a gradient function to a vector of neuron states.
 * @param [in] state The neuron states.
 * @param [in,out] delta The neuron gradients.
 * @param [in] n The length of the input array.
 * @param [in] a The activation function.
 */
void
neural_gradient_array(const double *state, double *delta, const int n,
                      const int a)
{
    for (int i = 0; i < n; ++i) {
        delta[i] *= neural_gradient(a, state[i]);
    }
}
