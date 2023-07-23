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
 * @file neural_layer_args_test.cpp
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
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("NEURAL_LAYER_ARGS")
{
    /* Test initialisation */
    struct XCSF xcsf;
    rand_init();
    param_init(&xcsf, 10, 2, 1);
    param_set_random_state(&xcsf, 1);
    struct ArgsLayer args;
    layer_args_init(&args);
    // first layer args
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
    args.evolve_connect = true;
    args.evolve_eta = true;
    args.evolve_functions = true;
    args.max_neuron_grow = 1;
    args.n_max = 10;
    // second layer args
    args.next = layer_args_copy(&args);
    args.next->type = DROPOUT;
    args.next->probability = 0.5;
    // third layer args
    args.next->next = layer_args_copy(&args);
    args.next->next->type = CONNECTED;
    layer_args_validate(&args);

    /* Smoke test export and import */
    char *json_str = layer_args_json_export(&args);
    CHECK(json_str != NULL);
    cJSON *json = cJSON_Parse(json_str);
    json_str = layer_args_json_import(&args, json->child);
    CHECK(json_str != NULL);

    /* Test clean up */
    param_free(&xcsf);
}
