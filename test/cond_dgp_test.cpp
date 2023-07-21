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
 * @file cond_dgp_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief DGP condition tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/cond_dgp.h"
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

TEST_CASE("COND_DGP")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Cl c1;
    struct Cl c2;
    rand_init();
    param_init(&xcsf, 5, 1, 1);
    param_set_random_state(&xcsf, 1);
    cond_param_set_type(&xcsf, COND_TYPE_DGP);
    cl_init(&xcsf, &c1, 1, 1);
    cl_init(&xcsf, &c2, 1, 1);
    cond_dgp_init(&xcsf, &c1);
    cond_dgp_init(&xcsf, &c2);

    /* Test covering */
    cond_dgp_cover(&xcsf, &c2, x);
    bool match = cond_dgp_match(&xcsf, &c2, x);
    CHECK_EQ(match, true);

    /* Test update */
    cond_dgp_update(&xcsf, &c1, x, y);

    /* Test copy */
    cond_dgp_free(&xcsf, &c2);
    cond_dgp_copy(&xcsf, &c2, &c1);
    struct CondDGP *src_cond = (struct CondDGP *) c1.cond;
    struct CondDGP *dest_cond = (struct CondDGP *) c2.cond;
    CHECK_EQ(dest_cond->dgp.n, src_cond->dgp.n);
    CHECK_EQ(dest_cond->dgp.klen, src_cond->dgp.klen);
    CHECK_EQ(dest_cond->dgp.max_t, src_cond->dgp.max_t);
    CHECK_EQ(dest_cond->dgp.max_k, src_cond->dgp.max_k);
    CHECK_EQ(dest_cond->dgp.n_inputs, src_cond->dgp.n_inputs);
    CHECK_EQ(dest_cond->dgp.t, src_cond->dgp.t);
    CHECK(check_array_eq(dest_cond->dgp.state, src_cond->dgp.state,
                         src_cond->dgp.n));
    CHECK(check_array_eq(dest_cond->dgp.initial_state,
                         src_cond->dgp.initial_state, src_cond->dgp.n));
    CHECK(check_array_eq_int(dest_cond->dgp.connectivity,
                             src_cond->dgp.connectivity, src_cond->dgp.klen));
    CHECK(check_array_eq(dest_cond->dgp.mu, src_cond->dgp.mu, 3));
    CHECK(check_array_eq_int(dest_cond->dgp.function, src_cond->dgp.function,
                             src_cond->dgp.n));

    /* Test size */
    CHECK_EQ(cond_dgp_size(&xcsf, &c1), src_cond->dgp.n);

    /* Test crossover */
    CHECK(!cond_dgp_crossover(&xcsf, &c1, &c2));

    /* Test general */
    CHECK(!cond_dgp_general(&xcsf, &c1, &c2));

    /* Test mutation */
    CHECK(cond_dgp_mutate(&xcsf, &c1));
    CHECK(!check_array_eq_int(dest_cond->dgp.connectivity,
                              src_cond->dgp.connectivity, src_cond->dgp.klen));

    /* Test import and export */
    char *json_str = cond_dgp_json_export(&xcsf, &c1);
    struct Cl new_cl;
    cl_init(&xcsf, &new_cl, 1, 1);
    cond_dgp_init(&xcsf, &new_cl);
    cJSON *json = cJSON_Parse(json_str);
    cond_dgp_json_import(&xcsf, &new_cl, json);
    struct CondDGP *orig_cond = (struct CondDGP *) c1.cond;
    struct CondDGP *new_cond = (struct CondDGP *) new_cl.cond;

    CHECK_EQ(new_cond->dgp.n, orig_cond->dgp.n);
    CHECK_EQ(new_cond->dgp.klen, orig_cond->dgp.klen);
    CHECK_EQ(new_cond->dgp.max_t, orig_cond->dgp.max_t);
    CHECK_EQ(new_cond->dgp.max_k, orig_cond->dgp.max_k);
    CHECK_EQ(new_cond->dgp.n_inputs, orig_cond->dgp.n_inputs);
    CHECK_EQ(new_cond->dgp.t, orig_cond->dgp.t);
    CHECK(check_array_eq(new_cond->dgp.state, orig_cond->dgp.state,
                         orig_cond->dgp.n));
    CHECK(check_array_eq(new_cond->dgp.initial_state,
                         orig_cond->dgp.initial_state, orig_cond->dgp.n));
    CHECK(check_array_eq_int(new_cond->dgp.connectivity,
                             orig_cond->dgp.connectivity, orig_cond->dgp.klen));
    CHECK(check_array_eq(new_cond->dgp.mu, orig_cond->dgp.mu, 3));
    CHECK(check_array_eq_int(new_cond->dgp.function, orig_cond->dgp.function,
                             orig_cond->dgp.n));

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = cond_dgp_save(&xcsf, &c1, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = cond_dgp_load(&xcsf, &c2, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* Test clean up */
    cond_dgp_free(&xcsf, &c1);
    cond_dgp_free(&xcsf, &c2);
}
