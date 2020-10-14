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
 * @date 2020.
 * @brief Loss function tests.
 */

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
    param_set_loss_func(&xcsf, LOSS_ONEHOT);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_ONEHOT);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 1);

    // Test HUBER
    param_set_loss_func(&xcsf, LOSS_HUBER);
    param_set_huber_delta(&xcsf, 1);
    CHECK_EQ(xcsf.LOSS_FUNC, LOSS_HUBER);
    CHECK_EQ(xcsf.HUBER_DELTA, 1);
    error = (xcsf.loss_ptr)(&xcsf, p, y);
    CHECK_EQ(doctest::Approx(error), 0.07132315);
}
