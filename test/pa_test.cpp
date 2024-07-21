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
 * @file pa_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2022--2024.
 * @brief Prediction array tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/pa.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("PA")
{
    struct XCSF xcsf;
    param_init(&xcsf, 5, 1, 5);
    param_set_random_state(&xcsf, 1);

    // test best action
    double pa1[5] = { 0.214, 0.6423, 0.111, 0.775, 0.445 };
    xcsf.pa = pa1;
    int action = pa_best_action(&xcsf);
    CHECK_EQ(action, 3);

    double pa2[5] = { 0.214, 0.9423, 0.111, 0.775, 0.445 };
    xcsf.pa = pa2;
    action = pa_best_action(&xcsf);
    CHECK_EQ(action, 1);

    double pa3[5] = { 0.6423, 0.6423, 0.6423, 0.6423, 0.445 };
    xcsf.pa = pa3;
    action = pa_best_action(&xcsf);
    CHECK_EQ(action, 2);

    action = pa_best_action(&xcsf);
    CHECK_EQ(action, 3);

    action = pa_best_action(&xcsf);
    CHECK_EQ(action, 2);

    param_free(&xcsf);
}
