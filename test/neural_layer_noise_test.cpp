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
 * @file neural_layer_softmax_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Noise neural network layer tests.
 */

#include "../lib/doctest/doctest/doctest.h"
#include <iostream>
#include <sstream>

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_noise.h"
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

TEST_CASE("NEURAL_LAYER_NOISE")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Net net;
    struct Layer *l;
    rand_init();
    param_init(&xcsf, 10, 2, 1);
    param_set_random_state(&xcsf, 2);
    pred_param_set_type(&xcsf, PRED_TYPE_NEURAL);
    neural_init(&net);
    struct ArgsLayer args;
    layer_args_init(&args);
    args.type = NOISE;
    args.n_inputs = 3;
    args.probability = 0.5;
    args.scale = 0.2;
    layer_args_validate(&args);
    l = layer_init(&args);
    neural_push(&net, l);
    CHECK_EQ(l->n_inputs, 3);
    CHECK_EQ(l->n_outputs, 3);
    CHECK_EQ(l->max_outputs, 3);
    CHECK_EQ(l->probability, 0.5);
    CHECK_EQ(l->scale, 0.2);

    const double x[3] = { 0.2, 0.5, 0.3 };

    /* Test one forward pass of input when predicting */
    net.train = false;
    const double output1[3] = { 0.2, 0.5, 0.3 };
    neural_layer_noise_forward(l, &net, x);
    double *out = neural_layer_noise_output(l);
    for (int i = 0; i < l->n_outputs; ++i) {
        CHECK_EQ(doctest::Approx(out[i]), output1[i]);
    }

    /* Test one forward pass of input when training */
    net.train = true;
    const double output2[3] = { 0.321, 0.5, 0.46388 };
    neural_layer_noise_forward(l, &net, x);
    out = neural_layer_noise_output(l);
    for (int i = 0; i < l->n_outputs; ++i) {
        CHECK_EQ(doctest::Approx(out[i]), output2[i]);
    }

    /* Test one backward pass of input */
    double delta[3] = { 0.2, 0.3, 0.4 };
    double delta_prev[3] = { 0.2, 0.3, 0.4 };
    neural_layer_noise_backward(l, &net, x, delta);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(doctest::Approx(delta[i]), delta_prev[i]);
    }

    /* Test update */
    neural_layer_noise_update(l);

    /* Test copy */
    struct Layer *l2 = neural_layer_noise_copy(l);
    CHECK_EQ(l->type, l2->type);
    CHECK_EQ(l->options, l2->options);
    CHECK_EQ(l->out_w, l2->out_w);
    CHECK_EQ(l->out_h, l2->out_h);
    CHECK_EQ(l->out_c, l2->out_c);
    CHECK_EQ(l->max_outputs, l2->max_outputs);
    CHECK_EQ(l->n_inputs, l2->n_inputs);
    CHECK_EQ(l->n_outputs, l2->n_outputs);
    CHECK_EQ(l->probability, l2->probability);
    CHECK_EQ(l->scale, l2->scale);

    /* Test mutate */
    CHECK_EQ(neural_layer_noise_mutate(l), false);

    /* Test randomisation */
    neural_layer_noise_rand(l);

    /* Smoke test export */
    char *json_str = neural_layer_noise_json_export(l, true);
    CHECK(json_str != NULL);

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = neural_layer_noise_save(l, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = neural_layer_noise_load(l, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* Test clean */
    neural_free(&net);
    param_free(&xcsf);
}
