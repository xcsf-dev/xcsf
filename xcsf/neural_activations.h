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
 * @file neural_activations.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2012--2020.
 * @brief Neural network activation functions.
 */

#pragma once

#include <math.h>

#define LOGISTIC (0) //!< Logistic [0,1]
#define RELU (1) //!< Rectified linear unit [0,inf]
#define TANH (2) //!< Tanh [-1,1]
#define LINEAR (3) //!< Linear [-inf,inf]
#define GAUSSIAN (4) //!< Gaussian (0,1]
#define SIN (5) //!< Sine [-1,1]
#define COS (6) //!< Cos [-1,1]
#define SOFT_PLUS (7) //!< Soft plus [0,inf]
#define LEAKY (8) //!< Leaky rectified linear unit [-inf,inf]
#define SELU (9) //!< Scaled-exponential linear unit [-1.7581,inf]
#define LOGGY (10) //!< Logistic [-1,1]
#define NUM_ACTIVATIONS (11) //!< Number of activations available
#define SOFT_MAX (100) //!< Softmax

#define STRING_LOGISTIC ("logistic\0") //!< Logistic
#define STRING_RELU ("relu\0") //!< RELU
#define STRING_TANH ("tanh\0") //!< Tanh
#define STRING_LINEAR ("linear\0") //!< Linear
#define STRING_GAUSSIAN ("gaussian\0") //!< Gaussian
#define STRING_SIN ("sin\0") //!< Sine
#define STRING_COS ("cos\0") //!< Cos
#define STRING_SOFT_PLUS ("softplus\0") //!< Soft plus
#define STRING_LEAKY ("leaky\0") //!< Leaky
#define STRING_SELU ("selu\0") //!< SELU
#define STRING_LOGGY ("loggy\0") //!< Loggy
#define STRING_SOFT_MAX ("softmax\0") //!< Softmax

double
neural_activate(const int a, const double x);

double
neural_gradient(const int a, const double x);

const char *
neural_activation_string(const int a);

int
neural_activation_as_int(const char *a);

void
neural_activate_array(double *state, double *output, const int n, const int a);

void
neural_gradient_array(const double *state, double *delta, const int n,
                      const int a);

static inline double
logistic_activate(const double x)
{
    return 1. / (1. + exp(-x));
}

static inline double
logistic_gradient(const double x)
{
    double fx = 1. / (1. + exp(-x));
    return (1 - fx) * fx;
}

static inline double
loggy_activate(const double x)
{
    return 2. / (1. + exp(-x)) - 1;
}

static inline double
loggy_gradient(const double x)
{
    double fx = exp(x);
    return (2 * fx) / ((fx + 1) * (fx + 1));
}

static inline double
gaussian_activate(const double x)
{
    return exp(-x * x);
}

static inline double
gaussian_gradient(const double x)
{
    return -2 * x * exp(-x * x);
}

static inline double
relu_activate(const double x)
{
    return x * (x > 0);
}

static inline double
relu_gradient(const double x)
{
    return (x > 0);
}

static inline double
selu_activate(const double x)
{
    return (x >= 0) * 1.0507 * x + (x < 0) * 1.0507 * 1.6732 * expm1(x);
}

static inline double
selu_gradient(const double x)
{
    return (x >= 0) * 1.0507 + (x < 0) * (1.0507 * 1.6732 * exp(x));
}

static inline double
linear_activate(const double x)
{
    return x;
}

static inline double
linear_gradient(const double x)
{
    (void) x;
    return 1;
}

static inline double
soft_plus_activate(const double x)
{
    return log1p(exp(x));
}

static inline double
soft_plus_gradient(const double x)
{
    return 1. / (1. + exp(-x));
}

static inline double
tanh_activate(const double x)
{
    return tanh(x);
}

static inline double
tanh_gradient(const double x)
{
    double t = tanh(x);
    return 1 - t * t;
}

static inline double
leaky_activate(const double x)
{
    return (x > 0) ? x : .1 * x;
}

static inline double
leaky_gradient(const double x)
{
    return (x < 0) ? .1 : 1;
}

static inline double
sin_activate(const double x)
{
    return sin(x);
}

static inline double
sin_gradient(const double x)
{
    return cos(x);
}

static inline double
cos_activate(const double x)
{
    return cos(x);
}

static inline double
cos_gradient(const double x)
{
    return -sin(x);
}
