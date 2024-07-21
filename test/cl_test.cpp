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
 * @file cl_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023--2024.
 * @brief Classifier tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/cond_rectangle.h"
#include "../xcsf/param.h"
#include "../xcsf/pred_nlms.h"
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

TEST_CASE("CL")
{
    /* Test initialisation */
    struct XCSF xcsf;
    struct Cl *c1 = (struct Cl *) malloc(sizeof(struct Cl));
    struct Cl *c2 = (struct Cl *) malloc(sizeof(struct Cl));
    rand_init();
    param_init(&xcsf, 5, 1, 1);
    param_set_random_state(&xcsf, 1);
    cl_init(&xcsf, c1, 1, 1);
    cl_init(&xcsf, c2, 1, 1);
    cl_rand(&xcsf, c1);

    /* Test copy condition, action, prediction */
    cl_copy(&xcsf, c2, c1);
    struct CondRectangle *cond1 = (struct CondRectangle *) c1->cond;
    struct CondRectangle *cond2 = (struct CondRectangle *) c2->cond;
    CHECK(check_array_eq(cond1->b1, cond2->b1, xcsf.x_dim));
    CHECK(check_array_eq(cond1->b2, cond2->b2, xcsf.x_dim));
    struct PredNLMS *pred1 = (struct PredNLMS *) c1->pred;
    struct PredNLMS *pred2 = (struct PredNLMS *) c2->pred;
    CHECK_EQ(pred1->n_weights, pred2->n_weights);
    CHECK_EQ(pred1->n, pred2->n);
    CHECK_EQ(pred1->eta, pred2->eta);
    CHECK(check_array_eq(pred1->weights, pred2->weights, pred1->n_weights));

    /* Test init and copy */
    struct Cl *dest = (struct Cl *) malloc(sizeof(struct Cl));
    cl_init_copy(&xcsf, dest, c1);
    CHECK_EQ(c1->err, dest->err);
    CHECK_EQ(c1->fit, dest->fit);
    CHECK_EQ(c1->num, dest->num);
    CHECK_EQ(c1->exp, dest->exp);
    CHECK_EQ(c1->size, dest->size);
    CHECK_EQ(c1->time, dest->time);
    CHECK_EQ(c1->m, dest->m);
    CHECK_EQ(c1->action, dest->action);
    CHECK_EQ(c1->age, dest->age);
    CHECK_EQ(c1->mtotal, dest->mtotal);
    CHECK(check_array_eq(c1->prediction, dest->prediction, xcsf.y_dim));

    /* Test import and export */
    char *json_str = cl_json_export(&xcsf, c1, true, true, true);
    CHECK(json_str != NULL);
    struct Cl *new_cl = (struct Cl *) malloc(sizeof(struct Cl));
    cJSON *json = cJSON_Parse(json_str);
    cl_json_import(&xcsf, new_cl, json);
    CHECK_EQ(c1->err, new_cl->err);
    CHECK_EQ(c1->fit, new_cl->fit);
    CHECK_EQ(c1->num, new_cl->num);
    CHECK_EQ(c1->exp, new_cl->exp);
    CHECK_EQ(c1->size, new_cl->size);
    CHECK_EQ(c1->time, new_cl->time);
    CHECK_EQ(c1->m, new_cl->m);
    CHECK_EQ(c1->action, new_cl->action);
    CHECK_EQ(c1->age, new_cl->age);
    CHECK_EQ(c1->mtotal, new_cl->mtotal);
    CHECK(check_array_eq(c1->prediction, new_cl->prediction, xcsf.y_dim));

    /* Test serialization */
    FILE *fp = fopen("temp.bin", "wb");
    size_t w = cl_save(&xcsf, c1, fp);
    fclose(fp);
    fp = fopen("temp.bin", "rb");
    struct Cl *load_cl = (struct Cl *) malloc(sizeof(struct Cl));
    size_t r = cl_load(&xcsf, load_cl, fp);
    CHECK_EQ(w, r);
    fclose(fp);

    /* Test clean up */
    cl_free(&xcsf, c1);
    cl_free(&xcsf, c2);
    cl_free(&xcsf, dest);
    cl_free(&xcsf, new_cl);
    cl_free(&xcsf, load_cl);
    free(json_str);
    cJSON_Delete(json);
    param_free(&xcsf);
}
