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
 * @file neural_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Neural network tests.
 */ 

#include "../lib/doctest/doctest/doctest.h"

extern "C" {   
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "../xcsf/xcsf.h"
#include "../xcsf/utils.h"
#include "../xcsf/param.h"
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_connected.h"
}

TEST_CASE("NEURAL")
{
    /* test one forward pass of input through two layers*/
    XCSF xcsf;
    NET net;
    LAYER *l;
    random_init();
    param_init(&xcsf);
    param_set_x_dim(&xcsf, 10);
    param_set_y_dim(&xcsf, 2);
    param_set_pred_type(&xcsf, PRED_TYPE_NEURAL);
    param_set_pred_eta(&xcsf, 0.1);
    param_set_pred_momentum(&xcsf, 0.9);

    const double x[10] = { -0.4792173279, -0.2056298252, -0.1775459629,
        -0.0814486626, 0.0923277094, 0.2779675621, -0.3109822596,
        -0.6788371120, -0.0714929928, -0.1332985280 };

    const double output[2] = { 0.5804315660, 0.2146193788 };

    const double orig_weights1[20] = { 0.3326639519, -0.4446678553,
        0.1033557369, -1.2581317787, 2.8042169798, 0.2236021733,
        -1.2206964138, -0.2022042865, -1.5489524535, -2.0932767781,
        5.4797621223, 0.3326639519, -0.4446678553, 0.1033557369,
        -1.2581317787, 2.8042169798, 0.2236021733, -1.2206964138,
        -0.2022042865, -1.5489524535 };

    const double orig_biases1[2] = { 0.1033557369, -1.2581317787 };

    const double orig_weights2[4] = { 0.3326639519, -0.4446678553,
        0.1033557369, -1.2581317787 };

    const double orig_biases2[2] = { 0.1033557369, -1.2581317787 };

    neural_init(&xcsf, &net);
    uint32_t o = 0;
    o |= LAYER_SGD_WEIGHTS;
    o |= LAYER_EVOLVE_WEIGHTS;

    l = neural_layer_connected_init(&xcsf, 10, 2, 2, LOGISTIC, o);
    memcpy(l->weights, orig_weights1, l->n_weights * sizeof(double));
    memcpy(l->biases, orig_biases1, l->n_outputs * sizeof(double));
    neural_layer_insert(&xcsf, &net, l, 0);

    l = neural_layer_connected_init(&xcsf, 2, 2, 2, LOGISTIC, o);
    memcpy(l->weights, orig_weights2, l->n_weights * sizeof(double));
    memcpy(l->biases, orig_biases2, l->n_outputs * sizeof(double));
    neural_layer_insert(&xcsf, &net, l, 1);

    neural_propagate(&xcsf, &net, x);

    double output_error = 0;
    for(int i = 0; i < net.n_outputs; i++) {
        output_error += fabs(neural_output(&xcsf, &net, i) - output[i]);
    }
    REQUIRE(doctest::Approx(output_error) == 0);

    /* test convergence on one input */
    const double y[2] = { 0.7343893899, 0.2289711363 };

    for(int i = 0; i < 200; i++) {
        neural_propagate(&xcsf, &net, x);
        neural_learn(&xcsf, &net, y, x);
    }
    REQUIRE(doctest::Approx(neural_output(&xcsf, &net, 0)) == y[0]);
    REQUIRE(doctest::Approx(neural_output(&xcsf, &net, 1)) == y[1]);
}
