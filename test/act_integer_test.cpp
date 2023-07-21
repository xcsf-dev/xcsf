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
 * @file act_integer_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Integer action tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/act_integer.h"
#include "../xcsf/action.h"
#include "../xcsf/cl.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("ACT_INTEGER")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Cl c1;
    struct Cl c2;
    rand_init();
    param_init(&xcsf, 3, 1, 10);
    param_set_random_state(&xcsf, 1);
    action_param_set_type(&xcsf, ACT_TYPE_INTEGER);
    cl_init(&xcsf, &c1, 1, 1);
    cl_init(&xcsf, &c2, 1, 1);
    act_integer_init(&xcsf, &c1);
    act_integer_init(&xcsf, &c2);
    const double x[3] = { 0.8455260670, 0.7566081103, 0.3125093674 };
    const double y[1] = { 0.2423423433 };

    /* Test cover */
    act_integer_cover(&xcsf, &c1, x, 1);
    struct ActInteger *a = (struct ActInteger *) c1.act;
    CHECK_EQ(a->action, 1);

    /* Test compute */
    CHECK_EQ(act_integer_compute(&xcsf, &c1, x), 1);

    /* Test update */
    act_integer_update(&xcsf, &c1, x, y);
    CHECK_EQ(a->action, 1);

    /* Test copy */
    act_integer_free(&xcsf, &c2);
    act_integer_copy(&xcsf, &c2, &c1);
    struct ActInteger *src_act = (struct ActInteger *) c1.act;
    struct ActInteger *dest_act = (struct ActInteger *) c2.act;
    CHECK_EQ(src_act->action, dest_act->action);

    /* Test general */
    CHECK(act_integer_general(&xcsf, &c1, &c2));
    --(src_act->action);
    CHECK(!act_integer_general(&xcsf, &c1, &c2));

    /* Test mutate: note this depends on random seed */
    a->mu[0] = 1;
    int before = a->action;
    CHECK(act_integer_mutate(&xcsf, &c1));
    CHECK(before != a->action);
    before = a->action;
    CHECK(!act_integer_mutate(&xcsf, &c1));
    CHECK(before == a->action);

    /* Test crossover */
    CHECK(!act_integer_crossover(&xcsf, &c1, &c2));

    /* Test import and export */
    char *json_str = act_integer_json_export(&xcsf, &c1);
    struct Cl new_cl;
    cl_init(&xcsf, &new_cl, 1, 1);
    act_integer_init(&xcsf, &new_cl);
    cJSON *json = cJSON_Parse(json_str);
    act_integer_json_import(&xcsf, &new_cl, json);
    struct ActInteger *orig_act = (struct ActInteger *) c1.act;
    struct ActInteger *new_act = (struct ActInteger *) new_cl.act;
    CHECK_EQ(new_act->action, orig_act->action);

    /* Test clean up */
    act_integer_free(&xcsf, &c1);
    act_integer_free(&xcsf, &c2);
    param_free(&xcsf);
}
