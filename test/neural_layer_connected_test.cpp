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
 * @file neural_layer_connected_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020--2024.
 * @brief Fully-connected neural network layer tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_connected.h"
#include "../xcsf/param.h"
#include "../xcsf/prediction.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("NEURAL_LAYER_CONNECTED")
{
    /* Test initialisation */
    struct XCSF xcsf;
    param_init(&xcsf, 10, 2, 1);
    param_set_random_state(&xcsf, 1);
    pred_param_set_type(&xcsf, PRED_TYPE_NEURAL);

    struct Net net;
    neural_init(&net);

    struct ArgsLayer args;
    layer_args_init(&args);
    args.type = CONNECTED;
    args.function = LOGISTIC;
    args.n_inputs = 10;
    args.n_init = 2;
    args.n_max = 2;
    args.eta = 0.1;
    args.momentum = 0.9;
    args.decay = 0;
    args.sgd_weights = true;
    layer_args_validate(&args);

    struct Layer *l = layer_init(&args);
    neural_push(&net, l);

    CHECK_EQ(l->function, LOGISTIC);
    CHECK_EQ(l->n_inputs, 10);
    CHECK_EQ(l->n_outputs, 2);
    CHECK_EQ(l->max_outputs, 2);
    CHECK_EQ(l->n_weights, 20);
    CHECK_EQ(l->eta, 0.1);
    CHECK_EQ(l->momentum, 0.9);

    /* Test one forward pass of input */
    const double x[10] = { -0.4792173279, -0.2056298252, -0.1775459629,
                           -0.0814486626, 0.0923277094,  0.2779675621,
                           -0.3109822596, -0.6788371120, -0.0714929928,
                           -0.1332985280 };
    const double output[2] = { 0.7936726123, 0.0963342482 };
    const double orig_weights[20] = {
        0.3326639519,  -0.4446678553, 0.1033557369,  -1.2581317787,
        2.8042169798,  0.2236021733,  -1.2206964138, -0.2022042865,
        -1.5489524535, -2.0932767781, 5.4797621223,  0.3326639519,
        -0.4446678553, 0.1033557369,  -1.2581317787, 2.8042169798,
        0.2236021733,  -1.2206964138, -0.2022042865, -1.5489524535
    };
    const double orig_biases[2] = { 0.1033557369, -1.2581317787 };

    memcpy(l->weights, orig_weights, sizeof(double) * l->n_weights);
    memcpy(l->biases, orig_biases, sizeof(double) * l->n_outputs);
    neural_layer_connected_forward(l, &net, x);
    double output_error = 0;
    for (int i = 0; i < l->n_outputs; ++i) {
        output_error += fabs(l->output[i] - output[i]);
    }
    CHECK_EQ(doctest::Approx(output_error), 0);

    /* Test one backward pass of input */
    const double y[2] = { 0.7343893899, 0.2289711363 };
    const double new_weights[20] = {
        0.3331291764,  -0.4444682297, 0.1035280986,  -1.2580527083,
        2.8041273480,  0.2233323222,  -1.2203945120, -0.2015452710,
        -1.5488830481, -2.0931473718, 5.4792087908,  0.3324265201,
        -0.4448728599, 0.1032616917,  -1.2580251719, 2.8045379369,
        0.2232430956,  -1.2214802376, -0.2022868364, -1.5491063675
    };
    const double new_biases[2] = { 0.1023849362, -1.2569771221 };
    for (int i = 0; i < l->n_outputs; ++i) {
        l->delta[i] = y[i] - l->output[i];
    }
    neural_layer_connected_backward(l, &net, x, 0);
    neural_layer_connected_update(l);
    double weight_error = 0;
    for (int i = 0; i < l->n_weights; ++i) {
        weight_error += fabs(l->weights[i] - new_weights[i]);
    }
    CHECK_EQ(doctest::Approx(weight_error), 0);
    double bias_error = 0;
    for (int i = 0; i < l->n_outputs; ++i) {
        bias_error += fabs(l->biases[i] - new_biases[i]);
    }
    CHECK_EQ(doctest::Approx(bias_error), 0);

    /* Test convergence on one input */
    const double conv_weights[20] = {
        0.4127301724,  -0.4103118294, 0.1330195938,  -1.2445235759,
        2.7887911379,  0.1771601713,  -1.1687384133, -0.0887861801,
        -1.5370076147, -2.0710056524, 5.2313215338,  0.2260593034,
        -0.5367129900, 0.0611303154,  -1.2102663341, 2.9483236736,
        0.0623796734,  -1.5726258658, -0.2392683912, -1.6180583952
    };
    const double conv_biases[2] = { -0.0637213195, -0.7397018847 };
    for (int i = 0; i < 200; ++i) {
        neural_layer_connected_forward(l, &net, x);
        for (int j = 0; j < l->n_outputs; ++j) {
            l->delta[j] = y[j] - l->output[j];
        }
        neural_layer_connected_backward(l, &net, x, 0);
        neural_layer_connected_update(l);
    }
    neural_layer_connected_forward(l, &net, x);
    CHECK_EQ(doctest::Approx(l->output[0]), y[0]);
    CHECK_EQ(doctest::Approx(l->output[1]), y[1]);
    double conv_weight_error = 0;
    for (int i = 0; i < l->n_weights; ++i) {
        conv_weight_error += fabs(l->weights[i] - conv_weights[i]);
    }
    CHECK_EQ(doctest::Approx(conv_weight_error), 0);
    double conv_bias_error = 0;
    for (int i = 0; i < l->n_outputs; ++i) {
        conv_bias_error += fabs(l->biases[i] - conv_biases[i]);
    }
    CHECK_EQ(doctest::Approx(conv_bias_error), 0);

    /* Test copy */
    struct Layer *l2 = neural_layer_connected_copy(l);
    CHECK_EQ(l->type, l2->type);
    CHECK_EQ(l->options, l2->options);
    CHECK_EQ(l->function, l2->function);
    CHECK_EQ(l->out_w, l2->out_w);
    CHECK_EQ(l->out_h, l2->out_h);
    CHECK_EQ(l->out_c, l2->out_c);
    CHECK_EQ(l->max_outputs, l2->max_outputs);
    CHECK_EQ(l->max_neuron_grow, l2->max_neuron_grow);
    CHECK_EQ(l->eta_max, l2->eta_max);
    CHECK_EQ(l->eta_min, l2->eta_min);
    CHECK_EQ(l->momentum, l2->momentum);
    CHECK_EQ(l->decay, l2->decay);
    CHECK_EQ(l->n_biases, l2->n_biases);
    CHECK_EQ(l->n_weights, l2->n_weights);
    CHECK_EQ(l->n_active, l2->n_active);
    CHECK_EQ(l->n_inputs, l2->n_inputs);
    CHECK_EQ(l->n_outputs, l2->n_outputs);
    CHECK_EQ(l->eta, l2->eta);
    for (int i = 0; i < l->n_weights; ++i) {
        CHECK(l->weights[i] == l2->weights[i]);
        CHECK(l->weight_active[i] == l2->weight_active[i]);
    }
    for (int i = 0; i < l->n_biases; ++i) {
        CHECK(l->biases[i] == l2->biases[i]);
    }
    for (int i = 0; i < 6; ++i) {
        CHECK(l->mu[i] == l2->mu[i]);
    }

    /* Test randomisation */
    neural_layer_connected_rand(l);
    for (int i = 0; i < l->n_weights; ++i) {
        CHECK(l->weights[i] != l2->weights[i]);
    }
    for (int i = 0; i < l->n_biases; ++i) {
        CHECK(l->biases[i] != l2->biases[i]);
    }

    /* Smoke test export */
    char *json_str = neural_layer_connected_json_export(l, true);
    CHECK(json_str != NULL);
    free(json_str);

    /* Clean up */
    neural_layer_connected_free(l2);
    free(l2);
    neural_free(&net);
    param_free(&xcsf);
}
