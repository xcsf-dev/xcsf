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
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_connected.h"
#include "../xcsf/neural_layer_lstm.h"
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

TEST_CASE("NEURAL_LAYER_LSTM")
{
    /* test initialisation */
    struct XCSF xcsf;
    struct Net net;
    struct Layer *l;
    rand_init();
    param_init(&xcsf, 1, 1, 1);
    pred_param_set_type(&xcsf, PRED_TYPE_NEURAL);
    neural_init(&net);
    struct ArgsLayer args;
    layer_args_init(&args);
    args.type = LSTM;
    args.function = TANH;
    args.recurrent_function = LOGISTIC;
    args.n_inputs = 1;
    args.n_init = 1;
    args.n_max = 1;
    args.eta = 0.1;
    args.momentum = 0.9;
    args.decay = 0;
    args.sgd_weights = true;
    l = layer_init(&args);
    neural_push(&net, l);
    CHECK_EQ(l->n_inputs, 1);
    CHECK_EQ(l->n_outputs, 1);
    CHECK_EQ(l->max_outputs, 1);
    CHECK_EQ(l->n_weights, 8);
    /* test forward passing input */
    const double x[1] = { 0.90598097 };
    const double orig_weights[8] = { 0.1866107,   -0.6872276,  1.0366809,
                                     -0.02821708, -0.21004653, 0.4503114,
                                     0.49545765,  0.71247584 };
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
    neural_layer_lstm_forward(l, &net, x);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.18687713);
    // second time
    neural_layer_lstm_forward(l, &net, x);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.30359772);
    // third time
    neural_layer_lstm_forward(l, &net, x);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.37268567);
    /* test one backward pass of input */
    const double y[1] = { 0.946146918 };
    for (int i = 0; i < l->n_outputs; ++i) {
        l->delta[i] = y[i] - l->output[i];
    }
    neural_layer_lstm_backward(l, &net, x, 0);
    neural_layer_lstm_update(l);
    // forward pass
    neural_layer_lstm_forward(l, &net, x);
    CHECK_EQ(doctest::Approx(l->output[0]), 0.4196390756);
    /* test convergence on one input */
    for (int i = 0; i < 400; ++i) {
        neural_layer_lstm_forward(l, &net, x);
        for (int j = 0; j < l->n_outputs; ++j) {
            l->delta[j] = y[j] - l->output[j];
        }
        neural_layer_lstm_backward(l, &net, x, 0);
        neural_layer_lstm_update(l);
    }
    neural_layer_lstm_forward(l, &net, x);
    CHECK_EQ(doctest::Approx(l->output[0]), y[0]);
    // clean up
    neural_free(&net);
    param_free(&xcsf);
}
