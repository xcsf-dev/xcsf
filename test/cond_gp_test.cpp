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
 * @file cond_gp_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief GP condition tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/cond_gp.h"
#include "../xcsf/condition.h"
#include "../xcsf/ea.h"
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

TEST_CASE("COND_GP")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Cl c1;
    struct Cl c2;
    rand_init();
    param_init(&xcsf, 5, 1, 1);
    param_set_random_state(&xcsf, 1);
    cond_param_set_type(&xcsf, COND_TYPE_GP);
    cl_init(&xcsf, &c1, 1, 1);
    cl_init(&xcsf, &c2, 1, 1);
    cond_gp_init(&xcsf, &c1);
    cond_gp_init(&xcsf, &c2);

    /* Test covering */
    cond_gp_cover(&xcsf, &c2, x);
    bool match = cond_gp_match(&xcsf, &c2, x);
    CHECK_EQ(match, true);

    /* Test update */
    cond_gp_update(&xcsf, &c1, x, y);

    /* Test copy */
    cond_gp_free(&xcsf, &c2);
    cond_gp_copy(&xcsf, &c2, &c1);
    struct CondGP *src_cond = (struct CondGP *) c1.cond;
    struct CondGP *dest_cond = (struct CondGP *) c2.cond;
    CHECK_EQ(dest_cond->gp.len, src_cond->gp.len);
    CHECK_EQ(dest_cond->gp.pos, src_cond->gp.pos);
    CHECK(check_array_eq(dest_cond->gp.mu, src_cond->gp.mu, 1));
    CHECK(check_array_eq_int(dest_cond->gp.tree, src_cond->gp.tree,
                             src_cond->gp.len));

    /* Test general */
    CHECK(!cond_gp_general(&xcsf, &c1, &c2));

    /* Test size */
    CHECK_EQ(cond_gp_size(&xcsf, &c1), src_cond->gp.len);

    /* Test crossover */
    ea_param_set_p_crossover(&xcsf, 1);
    CHECK(cond_gp_crossover(&xcsf, &c1, &c2));
    int n = src_cond->gp.len;
    if (src_cond->gp.len > dest_cond->gp.len) {
        n = dest_cond->gp.len;
    }
    CHECK(!check_array_eq_int(dest_cond->gp.tree, src_cond->gp.tree, n));

    /* Test mutation */
    cond_gp_free(&xcsf, &c2);
    cond_gp_copy(&xcsf, &c2, &c1);
    src_cond->gp.mu[0] = 1;
    CHECK(cond_gp_mutate(&xcsf, &c1));
    n = src_cond->gp.len;
    if (src_cond->gp.len > dest_cond->gp.len) {
        n = dest_cond->gp.len;
    }
    CHECK(!check_array_eq_int(dest_cond->gp.tree, src_cond->gp.tree, n));

    /* Smoke test export */
    CHECK(cond_gp_json_export(&xcsf, &c1) != NULL);

    /* Smoke test arg import and export */
    char *json_str = cond_gp_param_json_export(&xcsf);
    cJSON *json = cJSON_Parse(json_str);
    char *json_rtn = cond_gp_param_json_import(&xcsf, json->child);
    CHECK(json_rtn == NULL);

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = cond_gp_save(&xcsf, &c1, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = cond_gp_load(&xcsf, &c2, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* Test clean up */
    cond_gp_free(&xcsf, &c1);
    cond_gp_free(&xcsf, &c2);
}
