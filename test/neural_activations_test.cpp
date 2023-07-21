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
 * @file neural_activations_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Neural activation function tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/neural_activations.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("NEURAL_ACTIVATIONS")
{
    /* Test activations and gradients */
    const double x = 0.9;
    CHECK_EQ(neural_activate(LOGISTIC, x), doctest::Approx(0.71095));
    CHECK_EQ(neural_gradient(LOGISTIC, x), doctest::Approx(0.20550));
    CHECK_EQ(neural_activate(LOGGY, x), doctest::Approx(0.421899));
    CHECK_EQ(neural_gradient(LOGGY, x), doctest::Approx(0.411001));
    CHECK_EQ(neural_activate(GAUSSIAN, x), doctest::Approx(0.444858));
    CHECK_EQ(neural_gradient(GAUSSIAN, x), doctest::Approx(-0.800745));
    CHECK_EQ(neural_activate(RELU, x), doctest::Approx(0.9));
    CHECK_EQ(neural_gradient(RELU, x), doctest::Approx(1));
    CHECK_EQ(neural_activate(SELU, x), doctest::Approx(0.94563));
    CHECK_EQ(neural_gradient(SELU, x), doctest::Approx(1.0507));
    CHECK_EQ(neural_activate(LINEAR, x), doctest::Approx(0.9));
    CHECK_EQ(neural_gradient(LINEAR, x), doctest::Approx(1));
    CHECK_EQ(neural_activate(SOFT_PLUS, x), doctest::Approx(1.24115));
    CHECK_EQ(neural_gradient(SOFT_PLUS, x), doctest::Approx(0.71095));
    CHECK_EQ(neural_activate(TANH, x), doctest::Approx(0.716298));
    CHECK_EQ(neural_gradient(TANH, x), doctest::Approx(0.486917));
    CHECK_EQ(neural_activate(LEAKY, x), doctest::Approx(0.9));
    CHECK_EQ(neural_gradient(LEAKY, x), doctest::Approx(1));
    CHECK_EQ(neural_activate(SIN, x), doctest::Approx(0.783327));
    CHECK_EQ(neural_gradient(SIN, x), doctest::Approx(0.62161));
    CHECK_EQ(neural_activate(COS, x), doctest::Approx(0.62161));
    CHECK_EQ(neural_gradient(COS, x), doctest::Approx(-0.783327));

    /* Test string to int conversion */
    const int activations[NUM_ACTIVATIONS] = { LOGISTIC, RELU, TANH, LINEAR,
                                               GAUSSIAN, SIN,  COS,  SOFT_PLUS,
                                               LEAKY,    SELU, LOGGY };

    for (int i = 0; i < NUM_ACTIVATIONS; ++i) {
        const int a = activations[i];
        const char *str = neural_activation_string(a);
        const int integer = neural_activation_as_int(str);
        CHECK_EQ(integer, a);
    }

    /* Test array activation */
    const int x_dim = 4;
    double state[4] = { 0.1, 0.2, 0.3, 0.4 };
    double output[4] = { 0, 0, 0, 0 };
    double active[4] = { 0.524979, 0.549834, 0.574443, 0.598688 };
    neural_activate_array(state, output, x_dim, LOGISTIC);
    for (int i = 0; i < x_dim; ++i) {
        CHECK_EQ(output[i], doctest::Approx(active[i]));
    }

    /* Test array gradient */
    double delta[4] = { 1, 1, 1, 1 };
    double gradient[4] = { 0.249376, 0.247517, 0.244458, 0.240261 };
    neural_gradient_array(state, delta, x_dim, LOGISTIC);
    for (int i = 0; i < x_dim; ++i) {
        CHECK_EQ(delta[i], doctest::Approx(gradient[i]));
    }
}
