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
 * @file loss_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020--2024.
 * @brief Loss function tests.
 */

#include <cstdlib>
#include <stdexcept>

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/loss.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("LOSS FUNCTIONS")
{
    // Initialisation
    struct XCSF xcsf;
    rand_init();
    param_init(&xcsf, 10, 2, 1);
    param_set_random_state(&xcsf, 1);
    CHECK_EQ(xcsf.x_dim, 10);
    CHECK_EQ(xcsf.y_dim, 2);
    CHECK_EQ(xcsf.n_actions, 1);

    // Test input
    const double y[2] = { 0.7343893899, 0.2289711363 };
    const double p[2] = { 0.3334876345, -0.1239741663 };
    double error = 0;

    // Test MAE
    param_set_loss_func(&xcsf, LOSS_MAE);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_MAE);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 0.37692353);

    // Test MSE
    param_set_loss_func(&xcsf, LOSS_MSE);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_MSE);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 0.14264630);

    // Test RMSE
    param_set_loss_func(&xcsf, LOSS_RMSE);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_RMSE);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 0.37768545);

    // Test LOG
    param_set_loss_func(&xcsf, LOSS_LOG);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_LOG);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 9.05942561);

    // Test BINARY LOG
    param_set_loss_func(&xcsf, LOSS_BINARY_LOG);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_BINARY_LOG);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 9.07707219);

    // Test ONEHOT
    const double yh[2] = { 1, 0 };
    const double phc[2] = { 0.9, 0.1 };
    const double phw[2] = { 0.3, 0.7 };
    param_set_loss_func(&xcsf, LOSS_ONEHOT);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_ONEHOT);
    error = (xcsf.loss_ptr)(&xcsf, phw, yh);
    CHECK_EQ(doctest::Approx(error), 1); // incorrect
    error = (xcsf.loss_ptr)(&xcsf, phc, yh);
    CHECK_EQ(doctest::Approx(error), 0); // correct

    // Test HUBER
    param_set_loss_func(&xcsf, LOSS_HUBER);
    param_set_huber_delta(&xcsf, 1);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_HUBER);
    CHECK_EQ(xcsf.HUBER_DELTA, 1);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 0.07132315);

    // Test string to int conversion
    const int losses[LOSS_NUM] = { LOSS_MAE,  LOSS_MSE,        LOSS_RMSE,
                                   LOSS_LOG,  LOSS_BINARY_LOG, LOSS_ONEHOT,
                                   LOSS_HUBER };
    for (int i = 0; i < LOSS_NUM; ++i) {
        const int loss = losses[i];
        const char *str = loss_type_as_string(loss);
        const int integer = loss_type_as_int(str);
        CHECK_EQ(integer, loss);
    }

    // Test invalid loss type
    xcsf.LOSS_FUNC = 9999;
    int inv = loss_set_func(&xcsf);
    CHECK_EQ(inv, LOSS_INVALID);

    inv = loss_type_as_int("jsfsdf");
    CHECK_EQ(inv, LOSS_INVALID);

    // loss_type_as_string(LOSS_INVALID);

    // Clean up
    param_free(&xcsf);
}
