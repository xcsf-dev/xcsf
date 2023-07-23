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
 * @file neural_layer_convolutional_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020--2023.
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
#include "../xcsf/prediction.h"
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
    /* Test initialisation */
    struct XCSF xcsf;
    struct Net net;
    struct Layer *l;
    rand_init();
    param_init(&xcsf, 10, 2, 1);
    param_set_random_state(&xcsf, 1);
    pred_param_set_type(&xcsf, PRED_TYPE_NEURAL);
    neural_init(&net);
    struct ArgsLayer args;
    layer_args_init(&args);
    args.type = CONVOLUTIONAL;
    args.function = RELU;
    args.width = 4;
    args.height = 4;
    args.channels = 1;
    args.n_init = 2;
    args.size = 3;
    args.stride = 1;
    args.pad = 1;
    args.decay = 0;
    args.eta = 0.1;
    args.momentum = 0.9;
    args.sgd_weights = true;
    args.evolve_neurons = true;
    args.max_neuron_grow = 1;
    args.n_max = 10;
    layer_args_validate(&args);
    l = layer_init(&args);
    neural_push(&net, l);
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
    CHECK_EQ(l->momentum, 0.9);

    /* Test one forward pass of input */
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
                const int pos = j + l->size * (k + l->size * i);
                l->weights[pos] = orig_weights[index];
                ++index;
            }
        }
    }
    memcpy(l->biases, orig_biases, sizeof(double) * l->n_filters);
    neural_layer_convolutional_forward(l, &net, x);
    double output_error = 0;
    index = 0;

    const double *out = neural_layer_convolutional_output(l);
    for (int k = 0; k < l->out_h; ++k) {
        for (int j = 0; j < l->out_w; ++j) {
            for (int i = 0; i < l->out_c; ++i) {
                const double layer_output_i =
                    out[j + l->out_w * (k + l->out_h * i)];
                output_error += fabs(layer_output_i - output[index]);
                ++index;
            }
        }
    }
    CHECK_EQ(doctest::Approx(output_error), 0);

    /* Test convergence on one input */
    const double y[32] = { 0.,         0.,         0.,         0.,
                           0.,         0.,         0.24233836, 0.21147227,
                           0.82006556, 0.68110734, 0.7897921,  0.,
                           0.16564375, 0.,         0.,         0.,
                           0.,         0.,         0.,         0.,
                           0.,         0.,         0.,         0.,
                           1.0699066,  0.69851404, 1.7120876,  0.5649568,
                           1.8013113,  0.2873966,  1.2277601,  0. };
    for (int e = 0; e < 2000; ++e) {
        neural_layer_convolutional_forward(l, &net, x);
        index = 0;
        for (int k = 0; k < l->out_h; ++k) {
            for (int j = 0; j < l->out_w; ++j) {
                for (int i = 0; i < l->out_c; ++i) {
                    const int pos = j + l->out_w * (k + l->out_h * i);
                    l->delta[pos] = y[index] - l->output[pos];
                    ++index;
                }
            }
        }
        neural_layer_convolutional_backward(l, &net, x, 0);
        neural_layer_convolutional_update(l);
    }
    neural_layer_convolutional_forward(l, &net, x);
    double conv_error = 0;
    index = 0;
    for (int k = 0; k < l->out_h; ++k) {
        for (int j = 0; j < l->out_w; ++j) {
            for (int i = 0; i < l->out_c; ++i) {
                const int pos = j + l->out_w * (k + l->out_h * i);
                const double error = y[index] - l->output[pos];
                conv_error += error * error;
                ++index;
            }
        }
    }
    conv_error /= l->n_outputs; // MSE
    CHECK_EQ(doctest::Approx(conv_error), 0);

    /* Test copy */
    struct Layer *l2 = neural_layer_convolutional_copy(l);
    CHECK_EQ(l->type, l2->type);
    CHECK_EQ(l->options, l2->options);
    CHECK_EQ(l->function, l2->function);
    CHECK_EQ(l->height, l2->height);
    CHECK_EQ(l->width, l2->height);
    CHECK_EQ(l->channels, l2->channels);
    CHECK_EQ(l->n_filters, l2->n_filters);
    CHECK_EQ(l->max_outputs, l2->max_outputs);
    CHECK_EQ(l->stride, l2->stride);
    CHECK_EQ(l->size, l2->size);
    CHECK_EQ(l->pad, l2->pad);
    CHECK_EQ(l->max_neuron_grow, l2->max_neuron_grow);
    CHECK_EQ(l->eta_max, l2->eta_max);
    CHECK_EQ(l->eta_min, l2->eta_min);
    CHECK_EQ(l->momentum, l2->momentum);
    CHECK_EQ(l->decay, l2->decay);
    CHECK_EQ(l->n_biases, l2->n_biases);
    CHECK_EQ(l->n_weights, l2->n_weights);
    CHECK_EQ(l->n_active, l2->n_active);
    CHECK_EQ(l->out_h, l2->out_h);
    CHECK_EQ(l->out_w, l2->out_w);
    CHECK_EQ(l->out_c, l2->out_c);
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
    neural_layer_convolutional_rand(l);
    for (int i = 0; i < l->n_weights; ++i) {
        CHECK(l->weights[i] != l2->weights[i]);
    }
    for (int i = 0; i < l->n_biases; ++i) {
        CHECK(l->biases[i] != l2->biases[i]);
    }

    /* Test mutation */
    args.evolve_functions = true;
    args.evolve_weights = true;
    args.evolve_connect = true;
    args.evolve_eta = true;
    l->options = layer_args_opt(&args);
    for (int i = 0; i < 6; ++i) {
        l->mu[i] = 1;
    }
    int func = l->function;
    double eta = l->eta;
    int n_filters = l->n_filters;
    int n_weights = l->n_weights;
    int n_biases = l->n_biases;
    double *wc = (double *) malloc(sizeof(double) * l->n_weights);
    double *bc = (double *) malloc(sizeof(double) * l->n_biases);
    memcpy(wc, l->weights, sizeof(double) * l->n_weights);
    memcpy(bc, l->biases, sizeof(double) * l->n_biases);
    CHECK(neural_layer_convolutional_mutate(l));
    int n = n_weights ? (n_weights < l->n_weights) : l->n_weights;
    for (int i = 0; i < n; ++i) {
        CHECK(l->weights[i] != wc[i]);
    }
    n = n_biases ? (n_biases < l->n_biases) : l->n_biases;
    for (int i = 0; i < n; ++i) {
        CHECK(l->biases[i] != bc[i]);
    }
    CHECK(l->eta != eta);
    CHECK(l->function != func);
    CHECK(l->n_filters != n_filters);

    /* Smoke test export */
    CHECK(neural_layer_convolutional_json_export(l, true) != NULL);

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = neural_layer_convolutional_save(l, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = neural_layer_convolutional_load(l, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* Test clean up */
    neural_free(&net);
    param_free(&xcsf);
}
