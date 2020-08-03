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
 * @date 2020.
 * @brief Convolutional neural network layer tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_convolutional.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("NEURAL_LAYER_CONVOLUTIONAL")
{
    /* test initialisation */
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
    neural_init(&xcsf, &net);
    uint32_t o = 0;
    o |= LAYER_SGD_WEIGHTS;
    l = neural_layer_convolutional_init(&xcsf, 4, 4, 1, 2, 3, 1, 1, RELU, o);
    neural_layer_insert(&xcsf, &net, l, 0);
    CHECK_EQ(l->function, RELU);
    CHECK_EQ(l->n_filters, 2);
    CHECK_EQ(l->size, 3);
    CHECK_EQ(l->out_h, 4);
    CHECK_EQ(l->out_w, 4);
    CHECK_EQ(l->out_c, 2);
    CHECK_EQ(l->pad, 1);
    CHECK_EQ(l->stride, 1);
    CHECK_EQ(l->n_inputs, 16);
    CHECK_EQ(l->n_outputs, 32);
    CHECK_EQ(l->n_weights, 18);
    CHECK_EQ(l->eta, 0.1);
    CHECK_EQ(l->options, o);
    /* test one forward pass of input */
    const double orig_weights[18] = { -0.3494757, 0.37103638,  0.43885502,
                                      0.11762521, 0.35432652,  0.17391846,
                                      0.46650133, -0.00751933, 0.01440367,
                                      0.3583322,  0.3935847,   0.10529158,
                                      0.28923538, -0.28357792, 0.14083597,
                                      0.2338815,  -0.46515846, -0.36625803 };
    const double orig_biases[2] = { 0, 0 };
    const double x[16] = { 0.00003019, 0.00263328, 0.04917052, 0.28910958,
                           0.59115183, 0.38058756, 0.08781348, 0.00530301,
                           0.00006084, 0.00017717, 0.00943315, 0.13314144,
                           0.50049726, 0.81313912, 0.8360666,  0.75973192 };
    const double output[32] = { 0.,         0.,         0.20314004, 0.,
                                0.23570573, 0.,         0.05324797, 0.07956585,
                                0.15918063, 0.25231227, 0.33003914, 0.14661954,
                                0.2434422,  0.0395971,  0.17221428, 0.08195485,
                                0.08660228, 0.,         0.,         0.02246629,
                                0.,         0.,         0.3267745,  0.02144092,
                                0.3273376,  0.26499897, 0.5776568,  0.3773253,
                                0.7416452,  0.39779976, 0.45610222, 0.2851106 };
    int index = 0;
    for (int k = 0; k < l->size; ++k) {
        for (int j = 0; j < l->size; ++j) {
            for (int i = 0; i < l->n_filters; ++i) {
                int pos = j + l->size * (k + l->size * i);
                l->weights[pos] = orig_weights[index];
                ++index;
            }
        }
    }
    memcpy(l->biases, orig_biases, l->n_filters * sizeof(double));
    neural_layer_convolutional_forward(&xcsf, l, x);
    double output_error = 0;
    index = 0;
    for (int k = 0; k < l->out_h; ++k) {
        for (int j = 0; j < l->out_w; ++j) {
            for (int i = 0; i < l->out_c; ++i) {
                double layer_output_i =
                    l->output[j + l->out_w * (k + l->out_h * i)];
                output_error += fabs(layer_output_i - output[index]);
                ++index;
            }
        }
    }
    CHECK_EQ(doctest::Approx(output_error), 0);
    /* test convergence on one input */
    const double y[32] = { 0.,         0.,         0.,         0.,
                           0.,         0.,         0.24233836, 0.21147227,
                           0.82006556, 0.68110734, 0.7897921,  0.,
                           0.16564375, 0.,         0.,         0.,
                           0.,         0.,         0.,         0.,
                           0.,         0.,         0.,         0.,
                           1.0699066,  0.69851404, 1.7120876,  0.5649568,
                           1.8013113,  0.2873966,  1.2277601,  0. };
    for (int e = 0; e < 2000; ++e) {
        neural_layer_convolutional_forward(&xcsf, l, x);
        index = 0;
        for (int k = 0; k < l->out_h; ++k) {
            for (int j = 0; j < l->out_w; ++j) {
                for (int i = 0; i < l->out_c; ++i) {
                    int pos = j + l->out_w * (k + l->out_h * i);
                    l->delta[pos] = y[index] - l->output[pos];
                    ++index;
                }
            }
        }
        neural_layer_convolutional_backward(&xcsf, l, x, 0);
        neural_layer_convolutional_update(&xcsf, l);
    }
    neural_layer_convolutional_forward(&xcsf, l, x);
    double conv_error = 0;
    index = 0;
    for (int k = 0; k < l->out_h; ++k) {
        for (int j = 0; j < l->out_w; ++j) {
            for (int i = 0; i < l->out_c; ++i) {
                int pos = j + l->out_w * (k + l->out_h * i);
                double error = y[index] - l->output[pos];
                conv_error += error * error;
                ++index;
            }
        }
    }
    conv_error /= l->n_outputs; // MSE
    CHECK_EQ(doctest::Approx(conv_error), 0);
}
