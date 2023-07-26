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
 * @file clset_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Set tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/clset.h"
#include "../xcsf/pa.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcs_supervised.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("CLSET")
{
    /* Test initialisation */
    const int x_dim = 4;
    const int y_dim = 1;

    struct XCSF xcsf;
    param_init(&xcsf, x_dim, y_dim, 1);
    param_set_random_state(&xcsf, 2);
    xcsf_init(&xcsf);
    clset_pset_init(&xcsf);

    /* Test insert */
    // insert a classifier
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "error", 0.1);
    cJSON_AddNumberToObject(json, "fitness", 0.2);
    cJSON_AddNumberToObject(json, "set_size", 10);
    cJSON_AddNumberToObject(json, "numerosity", 20);
    cJSON_AddNumberToObject(json, "experience", 50);
    cJSON_AddNumberToObject(json, "time", 100);
    cJSON_AddNumberToObject(json, "samples_seen", 100);
    cJSON_AddNumberToObject(json, "samples_matched", 50);
    cJSON_AddBoolToObject(json, "current_match", true);
    cJSON_AddNumberToObject(json, "current_action", 1);
    double pred[1] = { 0.6 };
    cJSON *p = cJSON_CreateDoubleArray(pred, xcsf.y_dim);
    cJSON_AddItemToObject(json, "current_prediction", p);
    clset_json_insert_cl(&xcsf, json);
    cJSON_Delete(json);
    // check the classifier
    struct Cl *c = xcsf.pset.list->cl;
    CHECK_EQ(c->err, 0.1);
    CHECK_EQ(c->fit, 0.2);
    CHECK_EQ(c->size, 10);
    CHECK_EQ(c->num, 20);
    CHECK_EQ(c->exp, 50);
    CHECK_EQ(c->time, 100);
    CHECK_EQ(c->age, 100);
    CHECK_EQ(c->mtotal, 50);
    CHECK(c->m);
    CHECK_EQ(c->action, 1);
    for (int i = 0; i < y_dim; ++i) {
        CHECK_EQ(pred[i], c->prediction[i]);
    }

    /* Test mean size calculations */
    const double csize = clset_mean_cond_size(&xcsf, &xcsf.pset);
    CHECK_EQ(csize, x_dim);
    const double psize = clset_mean_pred_size(&xcsf, &xcsf.pset);
    CHECK_EQ(psize, x_dim + 1);

    /* Smoke test export */
    char *json_str = clset_json_export(&xcsf, &xcsf.pset, true, true, true);
    CHECK(json_str != NULL);

    /* Smoke test import */
    clset_json_insert(&xcsf, json_str);

    /* Test clean up */
    xcsf_free(&xcsf);
    param_free(&xcsf);
}
