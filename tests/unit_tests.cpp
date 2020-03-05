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
 * @file unit_tests.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Main unit test loading function.
 */ 

#undef DOCTEST_CONFIG_DISABLE
#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#include "../lib/doctest/doctest/doctest.h"

namespace xcsf {
extern "C" {   
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "../xcsf/xcsf.h"
#include "../xcsf/utils.h"
#include "../xcsf/config.h"
#include "../xcsf/blas.h"
#include "../xcsf/neural_activations.h"
#include "../xcsf/neural.h"
#include "../xcsf/neural_layer.h"
#include "../xcsf/neural_layer_connected.h"
#include "../xcsf/cl.h"
#include "../xcsf/prediction.h"
#include "../xcsf/pred_nlms.h"
#include "../xcsf/pred_rls.h"
}
XCSF xcsf;
CL c;
}

#include "xcsf_test.h"
#include "neural_layer_connected_test.h"
#include "pred_nlms_test.h"
#include "pred_rls_test.h"

int main(int argc, char **argv)
{
    xcsf::random_init();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int res = context.run();
    if(context.shouldExit()) {
        return res;
    }
}
