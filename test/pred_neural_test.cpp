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
 * @file pred_neural_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Neural prediction tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/param.h"
#include "../xcsf/pred_neural.h"
#include "../xcsf/prediction.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("PRED_NEURAL")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Cl c;
    param_init(&xcsf, 10, 1, 1);
    param_set_random_state(&xcsf, 2);
    xcsf_init(&xcsf);
    pred_param_set_type(&xcsf, PRED_TYPE_NEURAL);
    cl_init(&xcsf, &c, 1, 1);
    pred_neural_init(&xcsf, &c);

    /* Test compute */
    const double x[10] = { -0.4792173279, -0.2056298252, -0.1775459629,
                           -0.0814486626, 0.0923277094,  0.2779675621,
                           -0.3109822596, -0.6788371120, -0.0714929928,
                           -0.1332985280 };

    pred_neural_compute(&xcsf, &c, x);

    /* Test update */
    const double y[1] = { -0.8289711363 };
    pred_neural_update(&xcsf, &c, x, y);

    /* Test copy */
    struct Cl dest_cl;
    cl_init(&xcsf, &dest_cl, 1, 1);
    pred_neural_copy(&xcsf, &dest_cl, &c);

    /* Test print */
    CAPTURE(pred_neural_print(&xcsf, &c));

    /* Test crossover */
    CHECK(!pred_neural_crossover(&xcsf, &c, &dest_cl));

    /* Test mutation */
    CHECK(pred_neural_mutate(&xcsf, &c));

    /* test size */
    CHECK_EQ(pred_neural_size(&xcsf, &c), 108);

    /* Test n layers */
    CHECK_EQ(pred_neural_layers(&xcsf, &c), 2);

    /* Test n neurons */
    CHECK_EQ(pred_neural_neurons(&xcsf, &c, 0), 10);

    /* Test n connections */
    CHECK_EQ(pred_neural_connections(&xcsf, &c, 0), 98);

    /* Test eta */
    CHECK_EQ(pred_neural_eta(&xcsf, &c, 0), doctest::Approx(0.00685845));

    /* Test export */
    char *json_str = pred_neural_json_export(&xcsf, &c);
    CHECK(json_str != NULL);
    free(json_str);

    /* Test import -- not yet implemented */

    /* Test param import */
    const char *param_str = "{"
                            "\"layer_0\": {"
                            "\"type\": \"connected\","
                            "\"activation\": \"relu\""
                            "},"
                            "\"layer_1\": {"
                            "\"type\": \"connected\","
                            "\"activation\": \"linear\""
                            "}"
                            "}";
    cJSON *json = cJSON_Parse(param_str);
    char *ret = pred_neural_param_json_import(&xcsf, json->child);
    CHECK(ret == NULL);
    free(ret);
    CHECK(xcsf.pred->largs->type == layer_type_as_int("connected"));
    CHECK(xcsf.pred->largs->function == neural_activation_as_int("relu"));
    CHECK(xcsf.pred->largs->next->type == layer_type_as_int("connected"));
    CHECK(xcsf.pred->largs->next->function ==
          neural_activation_as_int("linear"));

    /* Test save */
    FILE *fp = fopen("temp.bin", "wb");
    size_t s = pred_neural_save(&xcsf, &c, fp);
    fclose(fp);

    /* Test load */
    fp = fopen("temp.bin", "rb");
    size_t r = pred_neural_load(&xcsf, &c, fp);
    CHECK_EQ(s, r);

    /* Test expand */
    pred_neural_expand(&xcsf, &c);

    /* Clean up */
    xcsf_free(&xcsf);
    param_free(&xcsf);
}
