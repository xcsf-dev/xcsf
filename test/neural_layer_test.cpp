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
 * @file neural_layer_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Neural network layer parameter tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_args.h"
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

#define N_TYPES 10

TEST_CASE("NEURAL_LAYER")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Net net;
    struct Layer *l;
    rand_init();
    param_init(&xcsf, 10, 2, 2);
    param_set_random_state(&xcsf, 1);
    pred_param_set_type(&xcsf, PRED_TYPE_NEURAL);
    neural_init(&net);
    struct ArgsLayer args;
    layer_args_init(&args);
    args.type = CONNECTED;
    args.function = LOGISTIC;
    args.n_inputs = 10;
    args.n_init = 2;
    args.n_max = 10;
    args.max_neuron_grow = 1;
    args.eta = 0.1;
    args.momentum = 0.9;
    args.decay = 0;
    args.sgd_weights = true;
    args.evolve_neurons = true;
    layer_args_validate(&args);
    xcsf_init(&xcsf);
    l = layer_init(&args);

    /* Test ensuring representation */
    l->n_active = 0;
    for (int i = 0; i < l->n_weights; ++i) {
        l->weight_active[i] = false;
    }
    layer_ensure_input_represention(l);
    CHECK(l->n_active != 0);

    /* Test add neurons */
    int n_neurons = l->n_outputs;
    int n_inputs = l->n_inputs;
    int n_active = l->n_active;
    layer_add_neurons(l, 1);
    CHECK_EQ(l->n_inputs, n_inputs);
    CHECK_EQ(l->n_outputs, n_neurons + 1);
    CHECK_EQ(l->n_biases, n_neurons + 1);
    CHECK_EQ(l->n_weights, n_inputs * (n_neurons + 1));
    CHECK(l->n_active > n_active);

    /* Test mutate neurons */
    int n_mutate = layer_mutate_neurons(l, 1);
    CHECK(n_mutate != 0);

    /* Test string to int conversion */
    const int types[N_TYPES] = { CONNECTED, DROPOUT, NOISE,   SOFTMAX,
                                 RECURRENT, LSTM,    MAXPOOL, CONVOLUTIONAL,
                                 AVGPOOL,   UPSAMPLE };

    for (int i = 0; i < N_TYPES; ++i) {
        const int t = types[i];
        const char *str = layer_type_as_string(t);
        const int integer = layer_type_as_int(str);
        CHECK_EQ(integer, t);
    }

    /* Test clean up */
    xcsf_free(&xcsf);
}
