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
 * @file neural_layer_maxpool_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Maxpooling neural network layer tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_maxpool.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("NEURAL_LAYER_MAXPOOL")
{
    /* test initialisation */
    struct Layer *l;
    rand_init();
    struct ArgsLayer args;
    layer_args_init(&args);
    args.type = MAXPOOL;
    args.width = 28;
    args.height = 28;
    args.channels = 1;
    args.size = 2;
    args.stride = 2;
    args.pad = 0;
    l = layer_init(&args);
    CHECK_EQ(l->width, 28);
    CHECK_EQ(l->height, 28);
    CHECK_EQ(l->channels, 1);
    CHECK_EQ(l->size, 2);
    CHECK_EQ(l->stride, 2);
    CHECK_EQ(l->pad, 0);
    CHECK_EQ(l->out_w, 14);
    CHECK_EQ(l->out_h, 14);
    CHECK_EQ(l->out_c, 1);
    CHECK_EQ(l->n_inputs, 784);
    CHECK_EQ(l->n_outputs, 196);
    layer_free(l);
}
