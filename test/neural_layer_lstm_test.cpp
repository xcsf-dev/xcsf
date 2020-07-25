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
 * @file neural_layer_lstm_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Long short-term memory layer tests.
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
#include "../xcsf/neural_layer_lstm.h"
}

TEST_CASE("NEURAL_LAYER_LSTM")
{
    /* test initialisation */
    XCSF xcsf;
    NET net;
    LAYER *l;
    random_init();
    param_init(&xcsf);
    param_set_x_dim(&xcsf, 1);
    param_set_y_dim(&xcsf, 1);
    param_set_pred_type(&xcsf, PRED_TYPE_NEURAL);
    param_set_pred_eta(&xcsf, 0.1);
    param_set_pred_momentum(&xcsf, 0.9);
    neural_init(&xcsf, &net);
    uint32_t o = 0;
    o |= LAYER_SGD_WEIGHTS;
    l = neural_layer_lstm_init(&xcsf, 1, 1, 1, TANH, LOGISTIC, o);
    neural_layer_insert(&xcsf, &net, l, 0);
    CHECK_EQ(l->n_inputs, 1);
    CHECK_EQ(l->n_outputs, 1);
    CHECK_EQ(l->max_outputs, 1);
    CHECK_EQ(l->options, o);
    CHECK_EQ(l->n_weights, 8);
    /* test forward passing input */
    const double x[1] = { 0.90598097 };
    const double orig_weights[8] = { 0.1866107, -0.6872276,  1.0366809,
        -0.02821708, -0.21004653,  0.4503114,  0.49545765,  0.71247584 };
    const double orig_biases[4] = { 0, 1, 0, 0 };
    l->ui->weights[0] = orig_weights[0];
    l->uf->weights[0] = orig_weights[1];
    l->ug->weights[0] = orig_weights[2];
    l->uo->weights[0] = orig_weights[3];
    l->wi->weights[0] = orig_weights[4];
    l->wf->weights[0] = orig_weights[5];
    l->wg->weights[0] = orig_weights[6];
    l->wo->weights[0] = orig_weights[7];
    l->ui->biases[0] = orig_biases[0];
    l->uf->biases[0] = orig_biases[1];
    l->ug->biases[0] = orig_biases[2];
    l->uo->biases[0] = orig_biases[3];
    l->wi->biases[0] = 0;
    l->wf->biases[0] = 0;
    l->wg->biases[0] = 0;
    l->wo->biases[0] = 0;
    // first time
    net.input = x;
    neural_layer_lstm_forward(&xcsf, l, &net);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.18687713);
    // second time
    net.input = x;
    neural_layer_lstm_forward(&xcsf, l, &net);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.30359772);
    // third time
    net.input = x;
    neural_layer_lstm_forward(&xcsf, l, &net);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.37268567);
    /* test one backward pass of input */
    const double y[1] = { 0.946146918 };
    net.input = x;
    net.delta = 0;
    for(int i = 0; i < l->n_outputs; i++) {
        l->delta[i] = y[i] - l->output[i];
    }
    neural_layer_lstm_backward(&xcsf, l, &net);
    neural_layer_lstm_update(&xcsf, l);
    // forward pass
    net.input = x;
    neural_layer_lstm_forward(&xcsf, l, &net);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.4196390756);
    /* test convergence on one input */
    for(int i = 0; i < 400; i++) {
        net.input = x;
        neural_layer_lstm_forward(&xcsf, l, &net);
        for(int j = 0; j < l->n_outputs; j++) {
            l->delta[j] = y[j] - l->output[j];
        }
        net.input = x;
        net.delta = 0;
        neural_layer_lstm_backward(&xcsf, l, &net);
        neural_layer_lstm_update(&xcsf, l);
    }
    net.input = x;
    neural_layer_lstm_forward(&xcsf, l, &net);
    CHECK_EQ(doctest::Approx(l->output[0]), y[0]);
}
