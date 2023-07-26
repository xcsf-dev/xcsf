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
 * @file cond_ternary_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020--2023.
 * @brief Ternary condition tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/cond_ternary.h"
#include "../xcsf/condition.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("COND_TERNARY")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Cl c1;
    rand_init();
    param_init(&xcsf, 5, 1, 1);
    param_set_random_state(&xcsf, 10);
    cond_param_set_type(&xcsf, COND_TYPE_TERNARY);
    cond_param_set_bits(&xcsf, 2);
    cl_init(&xcsf, &c1, 1, 1);
    cond_ternary_init(&xcsf, &c1);
    struct CondTernary *p = (struct CondTernary *) c1.cond;
    CHECK_EQ(p->length, 10);
    const double x[5] = { 0.8455260670, 0.0566081103, 0.3125093674,
                          0.3449376898, 0.5677518467 };
    const double y[1] = { 0.6 };

    /* test for true match condition */
    const char *true_1 = "1100010110";
    memcpy(p->string, true_1, sizeof(char) * 10);
    bool match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, true);
    const char *true_2 = "1#00#101#0";
    memcpy(p->string, true_2, sizeof(char) * 10);
    match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, true);

    /* test for false match condition */
    const char *false_1 = "1100000110";
    memcpy(p->string, false_1, sizeof(char) * 10);
    match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, false);
    const char *false_2 = "0#00#101#0";
    memcpy(p->string, false_2, sizeof(char) * 10);
    match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, false);

    /* test general */
    struct Cl c2;
    cl_init(&xcsf, &c2, 1, 1);
    cond_ternary_init(&xcsf, &c2);
    struct CondTernary *p2 = (struct CondTernary *) c2.cond;
    const char *spec = "0000#101#0";
    memcpy(p2->string, spec, sizeof(char) * 10);
    bool general = cond_ternary_general(&xcsf, &c1, &c2);
    CHECK_EQ(general, true);
    general = cond_ternary_general(&xcsf, &c2, &c1);
    CHECK_EQ(general, false);

    /* test covering */
    cond_ternary_cover(&xcsf, &c2, x);
    match = cond_ternary_match(&xcsf, &c2, x);
    CHECK_EQ(match, true);

    /* test size */
    double size = cond_ternary_size(&xcsf, &c1);
    CHECK_EQ(size, xcsf.cond->bits * xcsf.x_dim);

    /* test update */
    cond_ternary_update(&xcsf, &c1, x, y);

    /* test copy */
    struct Cl dest_cl;
    cl_init(&xcsf, &dest_cl, 1, 1);
    cond_ternary_copy(&xcsf, &dest_cl, &c1);
    struct CondTernary *dest_cond = (struct CondTernary *) dest_cl.cond;
    struct CondTernary *src_cond = (struct CondTernary *) c1.cond;
    CHECK_EQ(dest_cond->length, src_cond->length);
    for (int i = 0; i < src_cond->length; ++i) {
        CHECK_EQ(dest_cond->string[i], src_cond->string[i]);
    }
    for (int i = 0; i < 1; ++i) {
        CHECK_EQ(dest_cond->mu[i], src_cond->mu[i]);
    }

    /* test import and export */
    char *json_str = cond_ternary_json_export(&xcsf, &c1);
    CHECK(json_str != NULL);
    struct Cl new_cl;
    cl_init(&xcsf, &new_cl, 1, 1);
    cond_ternary_init(&xcsf, &new_cl);
    cJSON *json = cJSON_Parse(json_str);
    cond_ternary_json_import(&xcsf, &new_cl, json);
    struct CondTernary *new_cond = (struct CondTernary *) new_cl.cond;
    CHECK_EQ(new_cond->length, src_cond->length);
    for (int i = 0; i < src_cond->length; ++i) {
        CHECK_EQ(new_cond->string[i], src_cond->string[i]);
    }
    CHECK(check_array_eq(new_cond->mu, src_cond->mu, 1));
    free(json_str);
    cJSON_Delete(json);

    /* test mutation */
    CHECK(cond_ternary_mutate(&xcsf, &c1));
    bool equal = true;
    for (int i = 0; i < src_cond->length; ++i) {
        if (new_cond->string[i] != src_cond->string[i]) {
            equal = false;
        }
    }
    CHECK(!equal);

    /* test crossover */
    CHECK(cond_ternary_crossover(&xcsf, &c1, &c2));

    /* smoke test parameter import and export */
    json_str = cond_ternary_param_json_export(&xcsf);
    CHECK(json_str != NULL);

    json = cJSON_Parse(json_str);
    char *json_rtn = cond_ternary_param_json_import(&xcsf, json->child);
    CHECK(json_rtn == NULL);

    /* test serialisation */
    FILE *fp = fopen("temp.bin", "wb");
    size_t s = cond_ternary_save(&xcsf, &c1, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    size_t r = cond_ternary_load(&xcsf, &c1, fp);
    CHECK_EQ(s, r);

    /* clean up */
    cond_ternary_free(&xcsf, &c1);
    cond_ternary_free(&xcsf, &c2);
    param_free(&xcsf);
}
