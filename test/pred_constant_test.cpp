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
 * @file pred_nlms_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Constant prediction tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/param.h"
#include "../xcsf/pred_constant.h"
#include "../xcsf/prediction.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("PRED_CONSTANT")
{
    /* test initialisation */
    struct XCSF xcsf;
    struct Cl c1;
    struct Cl c2;
    param_init(&xcsf, 1, 1, 1);
    param_set_random_state(&xcsf, 1);
    param_set_beta(&xcsf, 0.005);
    xcsf_init(&xcsf);
    pred_param_set_type(&xcsf, PRED_TYPE_CONSTANT);
    cl_init(&xcsf, &c1, 1, 1);
    cl_init(&xcsf, &c2, 1, 1);

    /* Test init */
    pred_constant_init(&xcsf, &c1);
    pred_constant_init(&xcsf, &c2);

    /* test convergence on one input */
    const double x[4] = { 0.1, 0.2, 0.3, 0.4 };
    const double y[4] = { 0.1, 0.2, 0.3, 0.4 };
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 4; ++j) {
            ++(c1.exp);
            pred_constant_update(&xcsf, &c1, &x[j], &y[j]);
        }
    }
    CHECK_EQ(doctest::Approx(c1.prediction[0]), 0.25);

    /* test copy */
    pred_constant_copy(&xcsf, &c2, &c1);

    /* test print */
    CAPTURE(pred_constant_print(&xcsf, &c1));

    /* test crossover */
    CHECK(!pred_constant_crossover(&xcsf, &c1, &c2));

    /* test mutation */
    CHECK(!pred_constant_mutate(&xcsf, &c1));

    /* test size */
    CHECK_EQ(pred_constant_size(&xcsf, &c1), xcsf.y_dim);

    /* test import and export */
    char *json_str = pred_constant_json_export(&xcsf, &c1);
    struct Cl new_cl;
    cl_init(&xcsf, &new_cl, 1, 1);
    pred_constant_init(&xcsf, &new_cl);
    cJSON *json = cJSON_Parse(json_str);
    pred_constant_json_import(&xcsf, &new_cl, json);

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = pred_constant_save(&xcsf, &c1, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = pred_constant_load(&xcsf, &c2, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* clean up */
    param_free(&xcsf);
}
