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
 * @file cond_rectangle_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020--2024.
 * @brief Hyperrectangle condition tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/cond_rectangle.h"
#include "../xcsf/condition.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("COND_RECTANGLE_CSR")
{
    struct XCSF xcsf;
    param_init(&xcsf, 5, 1, 1);
    param_set_random_state(&xcsf, 1);
    xcsf_init(&xcsf);
    cond_param_set_type(&xcsf, COND_TYPE_HYPERRECTANGLE_CSR);
    cond_param_set_min(&xcsf, 0);
    cond_param_set_max(&xcsf, 1);
    cond_param_set_spread_min(&xcsf, 1);

    struct Cl *c1 = (struct Cl *) malloc(sizeof(struct Cl));
    cl_init(&xcsf, c1, 1, 1);
    cl_rand(&xcsf, c1);

    const double x[5] = { 0.8455260670, 0.7566081103, 0.3125093674,
                          0.3449376898, 0.3677518467 };
    const double true_center[5] = { 0.6917788795, 0.7276272381, 0.2457498699,
                                    0.2704867908, 0.0000000000 };
    const double true_spread[5] = { 0.5881265924, 0.8586376463, 0.2309959724,
                                    0.5802303236, 0.9674486498 };
    const double false_center[5] = { 0.8992419107, 0.5587937197, 0.6346787906,
                                     0.0464343089, 0.4214295062 };
    const double false_spread[5] = { 0.9658827122, 0.7107445754, 0.7048862747,
                                     0.1036188594, 0.4501471722 };

    /* test for true match condition */
    struct CondRectangle *p = (struct CondRectangle *) c1->cond;
    memcpy(p->b1, true_center, sizeof(double) * xcsf.x_dim);
    memcpy(p->b2, true_spread, sizeof(double) * xcsf.x_dim);
    bool match = cond_rectangle_match(&xcsf, c1, x);
    CHECK_EQ(match, true);

    /* test for false match condition */
    memcpy(p->b1, false_center, sizeof(double) * xcsf.x_dim);
    memcpy(p->b2, false_spread, sizeof(double) * xcsf.x_dim);
    match = cond_rectangle_match(&xcsf, c1, x);
    CHECK_EQ(match, false);

    /* test general */
    struct Cl *c2 = (struct Cl *) malloc(sizeof(struct Cl));
    cl_init(&xcsf, c2, 1, 1);
    cl_rand(&xcsf, c2);
    struct CondRectangle *p2 = (struct CondRectangle *) c2->cond;
    const double center2[5] = { 0.6, 0.7, 0.2, 0.3, 0.0 };
    const double spread2[5] = { 0.1, 0.1, 0.1, 0.1, 0.1 };
    memcpy(p2->b1, center2, sizeof(double) * xcsf.x_dim);
    memcpy(p2->b2, spread2, sizeof(double) * xcsf.x_dim);
    memcpy(p->b1, true_center, sizeof(double) * xcsf.x_dim);
    memcpy(p->b2, true_spread, sizeof(double) * xcsf.x_dim);
    bool general = cond_rectangle_general(&xcsf, c1, c2);
    CHECK_EQ(general, true);
    general = cond_rectangle_general(&xcsf, c2, c1);
    CHECK_EQ(general, false);

    /* test covering */
    cond_rectangle_cover(&xcsf, c2, x);
    match = cond_rectangle_match(&xcsf, c2, x);
    CHECK_EQ(match, true);

    /* test copy */
    cond_rectangle_free(&xcsf, c2);
    cond_rectangle_copy(&xcsf, c2, c1);
    struct CondRectangle *src_cond = (struct CondRectangle *) c1->cond;
    struct CondRectangle *dest_cond = (struct CondRectangle *) c2->cond;
    CHECK(check_array_eq(dest_cond->b1, src_cond->b1, xcsf.x_dim));
    CHECK(check_array_eq(dest_cond->b2, src_cond->b2, xcsf.x_dim));
    CHECK(check_array_eq(dest_cond->mu, src_cond->mu, 1));

    /* test size */
    CHECK_EQ(cond_rectangle_size(&xcsf, c1), xcsf.x_dim);

    /* test import and export */
    char *json_str = cond_rectangle_json_export(&xcsf, c1);
    struct Cl *new_cl = (struct Cl *) malloc(sizeof(struct Cl));
    cl_init(&xcsf, new_cl, 1, 1);
    cl_rand(&xcsf, new_cl);
    cJSON *json = cJSON_Parse(json_str);
    cond_rectangle_json_import(&xcsf, new_cl, json);
    struct CondRectangle *orig_cond = (struct CondRectangle *) c1->cond;
    struct CondRectangle *new_cond = (struct CondRectangle *) new_cl->cond;
    CHECK(check_array_eq(new_cond->b1, orig_cond->b1, xcsf.x_dim));
    CHECK(check_array_eq(new_cond->b2, orig_cond->b2, xcsf.x_dim));
    CHECK(check_array_eq(new_cond->mu, orig_cond->mu, 1));

    /* test mutation */
    CHECK(cond_rectangle_mutate(&xcsf, c1));
    CHECK(!check_array_eq(new_cond->b1, orig_cond->b1, xcsf.x_dim));
    CHECK(!check_array_eq(new_cond->b2, orig_cond->b2, xcsf.x_dim));

    /* test crossover */
    CHECK(cond_rectangle_crossover(&xcsf, c1, c2));

    /* test clean up */
    cl_free(&xcsf, c1);
    cl_free(&xcsf, c2);
    cl_free(&xcsf, new_cl);
    xcsf_free(&xcsf);
    param_free(&xcsf);
    free(json_str);
    cJSON_Delete(json);
}

TEST_CASE("COND_RECTANGLE_UBR")
{
    struct XCSF xcsf;
    param_init(&xcsf, 5, 1, 1);
    param_set_random_state(&xcsf, 1);
    xcsf_init(&xcsf);
    cond_param_set_type(&xcsf, COND_TYPE_HYPERRECTANGLE_UBR);
    cond_param_set_min(&xcsf, 0);
    cond_param_set_max(&xcsf, 1);
    cond_param_set_spread_min(&xcsf, 1);

    struct Cl *c1 = (struct Cl *) malloc(sizeof(struct Cl));
    cl_init(&xcsf, c1, 1, 1);
    cl_rand(&xcsf, c1);

    const double x[5] = { 0.8455260670, 0.7566081103, 0.3125093674,
                          0.3449376898, 0.3677518467 };
    const double true_b1[5] = { 0.8817788795, 0.7276272381, 0.2457498699,
                                0.2704867908, 0.0000000000 };
    const double true_b2[5] = { 0.5881265924, 0.8586376463, 0.3309959724,
                                0.5802303236, 0.9674486498 };
    const double false_b1[5] = { 0.8992419107, 0.5587937197, 0.6346787906,
                                 0.0464343089, 0.4214295062 };
    const double false_b2[5] = { 0.9658827122, 0.7107445754, 0.7048862747,
                                 0.1036188594, 0.4501471722 };

    /* test for true match condition */
    struct CondRectangle *p =
        reinterpret_cast<struct CondRectangle *>(c1->cond);
    memcpy(p->b1, true_b1, sizeof(double) * xcsf.x_dim);
    memcpy(p->b2, true_b2, sizeof(double) * xcsf.x_dim);
    bool match = cond_rectangle_match(&xcsf, c1, x);
    CHECK_EQ(match, true);

    /* test for false match condition */
    memcpy(p->b1, false_b1, sizeof(double) * xcsf.x_dim);
    memcpy(p->b2, false_b2, sizeof(double) * xcsf.x_dim);
    match = cond_rectangle_match(&xcsf, c1, x);
    CHECK_EQ(match, false);

    /* test general */
    struct Cl *c2 = (struct Cl *) malloc(sizeof(struct Cl));
    cl_init(&xcsf, c2, 1, 1);
    cl_rand(&xcsf, c2);
    struct CondRectangle *p2 =
        reinterpret_cast<struct CondRectangle *>(c2->cond);
    const double b1_2[5] = { 0.7, 0.8, 0.3, 0.3, 0.0 };
    const double b2_2[5] = { 0.6, 0.75, 0.31, 0.4, 0.1 };
    memcpy(p2->b1, b1_2, sizeof(double) * xcsf.x_dim);
    memcpy(p2->b2, b2_2, sizeof(double) * xcsf.x_dim);
    memcpy(p->b1, true_b1, sizeof(double) * xcsf.x_dim);
    memcpy(p->b2, true_b2, sizeof(double) * xcsf.x_dim);
    bool general = cond_rectangle_general(&xcsf, c1, c2);
    CHECK_EQ(general, true);
    general = cond_rectangle_general(&xcsf, c2, c1);
    CHECK_EQ(general, false);

    /* test covering */
    cond_rectangle_cover(&xcsf, c2, x);
    match = cond_rectangle_match(&xcsf, c2, x);
    CHECK_EQ(match, true);

    /* test clean up */
    cl_free(&xcsf, c1);
    cl_free(&xcsf, c2);
    xcsf_free(&xcsf);
    param_free(&xcsf);
}
