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
 * @file neural_layer_recurrent_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Recurrent neural network layer tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_connected.h"
#include "../xcsf/neural_layer_recurrent.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("NEURAL_LAYER_RECURRENT")
{
    /* test initialisation */
    struct XCSF xcsf;
    struct Net net;
    struct Layer *l;
    rand_init();
    param_init(&xcsf);
    param_set_x_dim(&xcsf, 1);
    param_set_y_dim(&xcsf, 1);
    param_set_pred_type(&xcsf, PRED_TYPE_NEURAL);
    param_set_pred_eta(&xcsf, 0.1);
    param_set_pred_momentum(&xcsf, 0.9);
    param_set_pred_decay(&xcsf, 0);
    neural_init(&xcsf, &net);
    uint32_t o = 0;
    o |= LAYER_SGD_WEIGHTS;
    o |= LAYER_EVOLVE_WEIGHTS;
    l = neural_layer_recurrent_init(&xcsf, 1, 1, 1, LOGISTIC, o);
    neural_push(&xcsf, &net, l);
    CHECK_EQ(l->function, LOGISTIC);
    CHECK_EQ(l->n_inputs, 1);
    CHECK_EQ(l->n_outputs, 1);
    CHECK_EQ(l->max_outputs, 1);
    CHECK_EQ(l->options, o);
    /* test forward passing input */
    const double x[1] = { 0.90598097 };
    const double orig_weights[2] = { -0.0735234, -1 };
    const double orig_biases[1] = { 0 };
    l->input_layer->weights[0] = orig_weights[0];
    l->input_layer->biases[0] = orig_biases[0];
    l->self_layer->weights[0] = orig_weights[1];
    l->self_layer->biases[0] = orig_biases[0];
    l->output_layer->weights[0] = 1;
    l->output_layer->biases[0] = 0;
    // first time
    neural_layer_recurrent_forward(&xcsf, l, x);
    double output_error = fabs(l->output[0] - 0.48335347);
    CHECK_EQ(doctest::Approx(output_error), 0);
    // second time
    neural_layer_recurrent_forward(&xcsf, l, x);
    output_error = fabs(l->output[0] - 0.3658727);
    CHECK_EQ(doctest::Approx(output_error), 0);
    // third time
    neural_layer_recurrent_forward(&xcsf, l, x);
    output_error = fabs(l->output[0] - 0.39353347);
    CHECK_EQ(doctest::Approx(output_error), 0);
    /* test one backward pass of input */
    const double y[1] = { 0.946146918 };
    for (int i = 0; i < l->n_outputs; ++i) {
        l->delta[i] = y[i] - l->output[i];
    }
    neural_layer_recurrent_backward(&xcsf, l, x, 0);
    neural_layer_recurrent_update(&xcsf, l);
    // forward pass
    neural_layer_recurrent_forward(&xcsf, l, x);
    output_error = fabs(l->output[0] - 0.3988695229);
    CHECK_EQ(doctest::Approx(output_error), 0);
    /* test convergence on one input */
    for (int i = 0; i < 400; ++i) {
        neural_layer_recurrent_forward(&xcsf, l, x);
        for (int j = 0; j < l->n_outputs; ++j) {
            l->delta[j] = y[j] - l->output[j];
        }
        neural_layer_recurrent_backward(&xcsf, l, x, 0);
        neural_layer_recurrent_update(&xcsf, l);
    }
    neural_layer_recurrent_forward(&xcsf, l, x);
    CHECK_EQ(doctest::Approx(l->output[0]), y[0]);
}
