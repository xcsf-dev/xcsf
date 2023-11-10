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
 * @file cond_neural_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Neural condition tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/cond_neural.h"
#include "../xcsf/condition.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

const double x[5] = { 0.8455260670, 0.7566081103, 0.3125093674, 0.3449376898,
                      0.3677518467 };

const double y[1] = { 0.9 };

TEST_CASE("COND_NEURAL")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Cl c1;
    struct Cl c2;
    param_init(&xcsf, 5, 1, 1);
    param_set_random_state(&xcsf, 1);
    xcsf_init(&xcsf);
    cond_param_set_type(&xcsf, COND_TYPE_NEURAL);
    cl_init(&xcsf, &c1, 1, 1);
    cl_init(&xcsf, &c2, 1, 1);
    condition_set(&xcsf, &c1);
    condition_set(&xcsf, &c2);

    /* Test init */
    cond_neural_init(&xcsf, &c1);
    cond_neural_init(&xcsf, &c2);

    /* Test covering */
    cond_neural_cover(&xcsf, &c2, x);
    bool match = cond_neural_match(&xcsf, &c2, x);
    CHECK_EQ(match, true);

    /* Test update */
    cond_neural_update(&xcsf, &c1, x, y);

    /* Test copy */
    cond_neural_free(&xcsf, &c2);
    cond_neural_copy(&xcsf, &c2, &c1);

    /* Test size */
    CHECK_EQ(cond_neural_size(&xcsf, &c1), 150);

    /* Test n layers */
    CHECK_EQ(cond_neural_layers(&xcsf, &c1), 2);

    /* Test n neurons */
    CHECK_EQ(cond_neural_neurons(&xcsf, &c1, 0), 10);

    /* Test n connections */
    CHECK_EQ(cond_neural_connections(&xcsf, &c1, 0), 50);

    /* Test crossover */
    CHECK(!cond_neural_crossover(&xcsf, &c1, &c2));

    /* Test general */
    CHECK(!cond_neural_general(&xcsf, &c1, &c2));

    /* Test mutation */
    CHECK(cond_neural_mutate(&xcsf, &c1));

    /* Test print */
    CAPTURE(cond_neural_print(&xcsf, &c1));

    /* Test export */
    char *json_str = cond_neural_json_export(&xcsf, &c1);
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
    char *ret = cond_neural_param_json_import(&xcsf, json->child);
    CHECK(ret == NULL);
    free(ret);
    CHECK(xcsf.cond->largs->type == layer_type_as_int("connected"));
    CHECK(xcsf.cond->largs->function == neural_activation_as_int("relu"));
    CHECK(xcsf.cond->largs->next->type == layer_type_as_int("connected"));
    CHECK(xcsf.cond->largs->next->function ==
          neural_activation_as_int("linear"));

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = cond_neural_save(&xcsf, &c1, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = cond_neural_load(&xcsf, &c2, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* Test clean up */
    cond_neural_free(&xcsf, &c1);
    cond_neural_free(&xcsf, &c2);
}
