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
#include "../xcsf/prediction.h"
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
    param_init(&xcsf, 1, 1, 1);
    param_set_random_state(&xcsf, 1);
    pred_param_set_type(&xcsf, PRED_TYPE_NEURAL);
    neural_init(&net);
    struct ArgsLayer args;
    layer_args_init(&args);
    args.type = RECURRENT;
    args.function = LOGISTIC;
    args.n_inputs = 1;
    args.n_init = 1;
    args.n_max = 1;
    args.eta = 0.1;
    args.momentum = 0.9;
    args.decay = 0;
    args.sgd_weights = true;
    args.evolve_neurons = true;
    args.evolve_connect = true;
    args.evolve_weights = true;
    args.evolve_functions = true;
    l = layer_init(&args);
    neural_push(&net, l);
    CHECK_EQ(l->function, LOGISTIC);
    CHECK_EQ(l->n_inputs, 1);
    CHECK_EQ(l->n_outputs, 1);
    CHECK_EQ(l->max_outputs, 1);

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
    neural_layer_recurrent_forward(l, &net, x);
    double output_error = fabs(l->output[0] - 0.48335347);
    CHECK_EQ(doctest::Approx(output_error), 0);
    // second time
    neural_layer_recurrent_forward(l, &net, x);
    output_error = fabs(l->output[0] - 0.3658727);
    CHECK_EQ(doctest::Approx(output_error), 0);
    // third time
    neural_layer_recurrent_forward(l, &net, x);
    output_error = fabs(l->output[0] - 0.39353347);
    CHECK_EQ(doctest::Approx(output_error), 0);

    /* test one backward pass of input */
    const double y[1] = { 0.946146918 };
    for (int i = 0; i < l->n_outputs; ++i) {
        l->delta[i] = y[i] - l->output[i];
    }
    neural_layer_recurrent_backward(l, &net, x, 0);
    neural_layer_recurrent_update(l);
    // forward pass
    neural_layer_recurrent_forward(l, &net, x);
    output_error = fabs(l->output[0] - 0.3988695229);
    CHECK_EQ(doctest::Approx(output_error), 0);

    /* Test convergence on one input */
    for (int i = 0; i < 400; ++i) {
        neural_layer_recurrent_forward(l, &net, x);
        for (int j = 0; j < l->n_outputs; ++j) {
            l->delta[j] = y[j] - l->output[j];
        }
        neural_layer_recurrent_backward(l, &net, x, 0);
        neural_layer_recurrent_update(l);
    }
    neural_layer_recurrent_forward(l, &net, x);
    CHECK_EQ(doctest::Approx(l->output[0]), y[0]);

    /* Test mutation */
    double *lw = (double *) malloc(sizeof(double) * l->input_layer->n_weights);
    memcpy(lw, l->input_layer->weights,
           sizeof(double) * l->input_layer->n_weights);
    CHECK(neural_layer_recurrent_mutate(l));
    for (int i = 0; i < l->input_layer->n_weights; ++i) {
        CHECK(l->input_layer->weights[i] != lw[i]);
    }

    /* Test copy */
    struct Layer *l2 = neural_layer_recurrent_copy(l);
    CHECK_EQ(l2->options, l->options);
    CHECK_EQ(l2->n_active, l->n_active);
    CHECK_EQ(l2->function, l->function);
    CHECK_EQ(l2->recurrent_function, l->recurrent_function);
    CHECK_EQ(l2->n_inputs, l->n_inputs);
    CHECK_EQ(l2->n_outputs, l->n_outputs);
    CHECK_EQ(l2->max_outputs, l->max_outputs);
    CHECK_EQ(l2->momentum, l->momentum);
    CHECK_EQ(l2->eta_max, l->eta_max);
    CHECK_EQ(l2->max_neuron_grow, l->max_neuron_grow);
    CHECK_EQ(l2->decay, l->decay);
    CHECK_EQ(l2->input_layer->n_biases, l->input_layer->n_biases);
    CHECK_EQ(l2->input_layer->n_weights, l->input_layer->n_weights);
    CHECK_EQ(l2->self_layer->n_biases, l->self_layer->n_biases);
    CHECK_EQ(l2->self_layer->n_weights, l->self_layer->n_weights);
    CHECK_EQ(l2->output_layer->n_biases, l->output_layer->n_biases);
    CHECK_EQ(l2->output_layer->n_weights, l->output_layer->n_weights);
    for (int i = 0; i < l->input_layer->n_weights; ++i) {
        CHECK(l->input_layer->weights[i] == l2->input_layer->weights[i]);
    }
    for (int i = 0; i < l->self_layer->n_weights; ++i) {
        CHECK(l->self_layer->weights[i] == l2->self_layer->weights[i]);
    }
    for (int i = 0; i < l->output_layer->n_weights; ++i) {
        CHECK(l->output_layer->weights[i] == l2->output_layer->weights[i]);
    }

    /* Test randomisation */
    neural_layer_recurrent_rand(l);
    for (int i = 0; i < l->input_layer->n_weights; ++i) {
        CHECK(l->input_layer->weights[i] != l2->input_layer->weights[i]);
    }
    for (int i = 0; i < l->self_layer->n_weights; ++i) {
        CHECK(l->self_layer->weights[i] != l2->self_layer->weights[i]);
    }
    for (int i = 0; i < l->output_layer->n_weights; ++i) {
        CHECK(l->output_layer->weights[i] != l2->output_layer->weights[i]);
    }

    /* Smoke test export */
    char *json_str = neural_layer_recurrent_json_export(l, true);
    CHECK(json_str != NULL);

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = neural_layer_recurrent_save(l, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = neural_layer_recurrent_load(l, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* Clean up */
    neural_free(&net);
    param_free(&xcsf);
}
