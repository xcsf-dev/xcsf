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
 * @file pred_rls_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Prediction module tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/param.h"
#include "../xcsf/prediction.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("PREDICTION")
{
    /* prediction type as str */
    const int MAX = 100;

    const char *ret = prediction_type_as_string(PRED_TYPE_CONSTANT);
    CHECK_EQ(strncmp(ret, PRED_STRING_CONSTANT, MAX), 0);

    ret = prediction_type_as_string(PRED_TYPE_NLMS_LINEAR);
    CHECK_EQ(strncmp(ret, PRED_STRING_NLMS_LINEAR, MAX), 0);

    ret = prediction_type_as_string(PRED_TYPE_NLMS_QUADRATIC);
    CHECK_EQ(strncmp(ret, PRED_STRING_NLMS_QUADRATIC, MAX), 0);

    ret = prediction_type_as_string(PRED_TYPE_RLS_LINEAR);
    CHECK_EQ(strncmp(ret, PRED_STRING_RLS_LINEAR, MAX), 0);

    ret = prediction_type_as_string(PRED_TYPE_RLS_QUADRATIC);
    CHECK_EQ(strncmp(ret, PRED_STRING_RLS_QUADRATIC, MAX), 0);

    ret = prediction_type_as_string(PRED_TYPE_NEURAL);
    CHECK_EQ(strncmp(ret, PRED_STRING_NEURAL, MAX), 0);

    /* prediction type as int */
    int type = prediction_type_as_int(PRED_STRING_CONSTANT);
    CHECK_EQ(type, PRED_TYPE_CONSTANT);

    type = prediction_type_as_int(PRED_STRING_NLMS_LINEAR);
    CHECK_EQ(type, PRED_TYPE_NLMS_LINEAR);

    type = prediction_type_as_int(PRED_STRING_NLMS_QUADRATIC);
    CHECK_EQ(type, PRED_TYPE_NLMS_QUADRATIC);

    type = prediction_type_as_int(PRED_STRING_RLS_LINEAR);
    CHECK_EQ(type, PRED_TYPE_RLS_LINEAR);

    type = prediction_type_as_int(PRED_STRING_RLS_QUADRATIC);
    CHECK_EQ(type, PRED_TYPE_RLS_QUADRATIC);

    type = prediction_type_as_int(PRED_STRING_NEURAL);
    CHECK_EQ(type, PRED_TYPE_NEURAL);
}
