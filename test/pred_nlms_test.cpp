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
 * @date 2020--2023.
 * @brief Normalised least mean squares unit tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/param.h"
#include "../xcsf/pred_nlms.h"
#include "../xcsf/prediction.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("PRED_NLMS")
{
    /* test initialisation */
    struct XCSF xcsf;
    struct Cl c;
    param_init(&xcsf, 10, 1, 1);
    param_set_random_state(&xcsf, 1);
    xcsf_init(&xcsf);
    pred_param_set_type(&xcsf, PRED_TYPE_NLMS_LINEAR);
    pred_param_set_x0(&xcsf, 1);
    pred_param_set_evolve_eta(&xcsf, false);
    pred_param_set_eta(&xcsf, 0.1);
    cl_init(&xcsf, &c, 1, 1);
    prediction_set(&xcsf, &c);
    pred_nlms_init(&xcsf, &c);
    struct PredNLMS *p = (struct PredNLMS *) c.pred;
    CHECK_EQ(p->n, 11);
    CHECK_EQ(p->n_weights, 11);

    /* test one forward pass of input */
    const double x[10] = { -0.4792173279, -0.2056298252, -0.1775459629,
                           -0.0814486626, 0.0923277094,  0.2779675621,
                           -0.3109822596, -0.6788371120, -0.0714929928,
                           -0.1332985280 };
    const double orig_weights[11] = {
        0.3326639519,  -0.4446678553, 0.1033557369,  -1.2581317787,
        2.8042169798,  0.2236021733,  -1.2206964138, -0.2022042865,
        -1.5489524535, -2.0932767781, 5.4797621223
    };
    memcpy(p->weights, orig_weights, sizeof(double) * 11);
    pred_nlms_compute(&xcsf, &c, x);
    CHECK_EQ(doctest::Approx(c.prediction[0]), 0.7343893899);

    /* test one backward pass of input */
    const double y[1] = { -0.8289711363 };
    const double new_weights[11] = {
        0.2535580953,  -0.4067589581, 0.1196222604,  -1.2440868532,
        2.8106600460,  0.2162985108,  -1.2426852759, -0.1776037685,
        -1.4952524623, -2.0876212637, 5.4903068165
    };
    pred_nlms_update(&xcsf, &c, x, y);
    double weight_error = 0;
    for (int i = 0; i < 11; ++i) {
        weight_error += fabs(p->weights[i] - new_weights[i]);
    }
    CHECK_EQ(doctest::Approx(weight_error), 0);

    /* test convergence on one input */
    for (int i = 0; i < 200; ++i) {
        pred_nlms_compute(&xcsf, &c, x);
        pred_nlms_update(&xcsf, &c, x, y);
    }
    pred_nlms_compute(&xcsf, &c, x);
    CHECK_EQ(doctest::Approx(c.prediction[0]), y[0]);

    /* test copy */
    struct Cl dest_cl;
    cl_init(&xcsf, &dest_cl, 1, 1);
    pred_nlms_copy(&xcsf, &dest_cl, &c);
    struct PredNLMS *dest_pred = (struct PredNLMS *) dest_cl.pred;
    struct PredNLMS *src_pred = (struct PredNLMS *) c.pred;
    CHECK_EQ(dest_pred->eta, src_pred->eta);
    CHECK_EQ(dest_pred->n, src_pred->n);
    CHECK_EQ(dest_pred->n_weights, src_pred->n_weights);
    CHECK(check_array_eq(dest_pred->weights, src_pred->weights,
                         src_pred->n_weights));

    /* test print */
    CAPTURE(pred_nlms_print(&xcsf, &c));

    /* test crossover */
    CHECK(!pred_nlms_crossover(&xcsf, &c, &dest_cl));

    /* test mutation */
    dest_pred->mu[0] = 0.1;
    dest_pred->eta = 0.01;
    xcsf.pred->evolve_eta = true;
    CHECK(pred_nlms_mutate(&xcsf, &dest_cl));

    /* test size */
    CHECK_EQ(pred_nlms_size(&xcsf, &c), src_pred->n_weights);

    /* test import and export */
    char *json_str = pred_nlms_json_export(&xcsf, &c);
    struct Cl new_cl;
    cl_init(&xcsf, &new_cl, 1, 1);
    pred_nlms_init(&xcsf, &new_cl);
    cJSON *json = cJSON_Parse(json_str);
    pred_nlms_json_import(&xcsf, &new_cl, json);
    struct PredNLMS *new_pred = (struct PredNLMS *) new_cl.pred;
    CHECK_EQ(new_pred->eta, src_pred->eta);
    CHECK_EQ(new_pred->n, src_pred->n);
    CHECK_EQ(new_pred->n_weights, src_pred->n_weights);
    CHECK(check_array_eq(new_pred->weights, src_pred->weights,
                         src_pred->n_weights));

    /* clean up */
    param_free(&xcsf);
}
