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
 * @date 2020.
 * @brief Normalised least mean squares unit tests.
 */ 

#include "../lib/doctest/doctest/doctest.h"

extern "C" {   
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "../xcsf/xcsf.h"
#include "../xcsf/utils.h"
#include "../xcsf/param.h"
#include "../xcsf/cl.h"
#include "../xcsf/prediction.h"
#include "../xcsf/pred_nlms.h"
}

TEST_CASE("PRED_NLMS")
{
    /* test initialisation */
    XCSF xcsf;
    CL c;
    random_init();
    param_init(&xcsf);
    param_set_x_dim(&xcsf, 10);
    param_set_y_dim(&xcsf, 1);
    param_set_pred_type(&xcsf, PRED_TYPE_NLMS_LINEAR);
    param_set_pred_evolve_eta(&xcsf, false);
    param_set_pred_x0(&xcsf, 1);
    param_set_pred_eta(&xcsf, 0.1);
    cl_init(&xcsf, &c, 1, 1);
    pred_nlms_init(&xcsf, &c);
    PRED_NLMS *p = (PRED_NLMS *) c.pred;
    CHECK_EQ(p->n, 11);
    CHECK_EQ(p->n_weights, 11);

    /* test one forward pass of input */
    const double x[10] = { -0.4792173279, -0.2056298252, -0.1775459629,
        -0.0814486626, 0.0923277094, 0.2779675621, -0.3109822596,
        -0.6788371120, -0.0714929928, -0.1332985280 };

    const double orig_weights[11] = { 0.3326639519, -0.4446678553,
        0.1033557369, -1.2581317787, 2.8042169798, 0.2236021733,
        -1.2206964138, -0.2022042865, -1.5489524535, -2.0932767781,
        5.4797621223 };

    memcpy(p->weights, orig_weights, 11 * sizeof(double));
    pred_nlms_compute(&xcsf, &c, x);
    CHECK_EQ(doctest::Approx(c.prediction[0]), 0.7343893899);

    /* test one backward pass of input */
    const double y[1] = { -0.8289711363 };

    const double new_weights[11] = { 0.2535580953, -0.4067589581,
        0.1196222604, -1.2440868532, 2.8106600460, 0.2162985108,
        -1.2426852759, -0.1776037685, -1.4952524623, -2.0876212637,
        5.4903068165 };

    pred_nlms_update(&xcsf, &c, x, y);
    double weight_error = 0;
    for(int i = 0; i < 11; i++) {
        weight_error += fabs(p->weights[i] - new_weights[i]);
    }
    CHECK_EQ(doctest::Approx(weight_error), 0);

    /* test convergence on one input */
    for(int i = 0; i < 200; i++) {
        pred_nlms_compute(&xcsf, &c, x);
        pred_nlms_update(&xcsf, &c, x, y);
    }
    pred_nlms_compute(&xcsf, &c, x);
    CHECK_EQ(doctest::Approx(c.prediction[0]), y[0]);
}
