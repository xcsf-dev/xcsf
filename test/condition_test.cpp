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
 * @brief Condition module tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/condition.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("CONDITION")
{
    /* condition type as str */
    const int MAX = 100;

    const char *ret = condition_type_as_string(COND_TYPE_DUMMY);
    CHECK_EQ(strncmp(ret, COND_STRING_DUMMY, MAX), 0);

    ret = condition_type_as_string(COND_TYPE_HYPERRECTANGLE_CSR);
    CHECK_EQ(strncmp(ret, COND_STRING_HYPERRECTANGLE_CSR, MAX), 0);

    ret = condition_type_as_string(COND_TYPE_HYPERRECTANGLE_UBR);
    CHECK_EQ(strncmp(ret, COND_STRING_HYPERRECTANGLE_UBR, MAX), 0);

    ret = condition_type_as_string(COND_TYPE_HYPERELLIPSOID);
    CHECK_EQ(strncmp(ret, COND_STRING_HYPERELLIPSOID, MAX), 0);

    ret = condition_type_as_string(COND_TYPE_NEURAL);
    CHECK_EQ(strncmp(ret, COND_STRING_NEURAL, MAX), 0);

    ret = condition_type_as_string(COND_TYPE_GP);
    CHECK_EQ(strncmp(ret, COND_STRING_GP, MAX), 0);

    ret = condition_type_as_string(COND_TYPE_DGP);
    CHECK_EQ(strncmp(ret, COND_STRING_DGP, MAX), 0);

    ret = condition_type_as_string(COND_TYPE_TERNARY);
    CHECK_EQ(strncmp(ret, COND_STRING_TERNARY, MAX), 0);

    ret = condition_type_as_string(RULE_TYPE_DGP);
    CHECK_EQ(strncmp(ret, COND_STRING_RULE_DGP, MAX), 0);

    ret = condition_type_as_string(RULE_TYPE_NEURAL);
    CHECK_EQ(strncmp(ret, COND_STRING_RULE_NEURAL, MAX), 0);

    ret = condition_type_as_string(RULE_TYPE_NETWORK);
    CHECK_EQ(strncmp(ret, COND_STRING_RULE_NETWORK, MAX), 0);

    /* condition type as int */
    int type = condition_type_as_int(COND_STRING_DUMMY);
    CHECK_EQ(type, COND_TYPE_DUMMY);

    type = condition_type_as_int(COND_STRING_HYPERRECTANGLE_CSR);
    CHECK_EQ(type, COND_TYPE_HYPERRECTANGLE_CSR);

    type = condition_type_as_int(COND_STRING_HYPERRECTANGLE_UBR);
    CHECK_EQ(type, COND_TYPE_HYPERRECTANGLE_UBR);

    type = condition_type_as_int(COND_STRING_HYPERELLIPSOID);
    CHECK_EQ(type, COND_TYPE_HYPERELLIPSOID);

    type = condition_type_as_int(COND_STRING_NEURAL);
    CHECK_EQ(type, COND_TYPE_NEURAL);

    type = condition_type_as_int(COND_STRING_GP);
    CHECK_EQ(type, COND_TYPE_GP);

    type = condition_type_as_int(COND_STRING_DGP);
    CHECK_EQ(type, COND_TYPE_DGP);

    type = condition_type_as_int(COND_STRING_TERNARY);
    CHECK_EQ(type, COND_TYPE_TERNARY);

    type = condition_type_as_int(COND_STRING_RULE_DGP);
    CHECK_EQ(type, RULE_TYPE_DGP);

    type = condition_type_as_int(COND_STRING_RULE_NEURAL);
    CHECK_EQ(type, RULE_TYPE_NEURAL);

    type = condition_type_as_int(COND_STRING_RULE_NETWORK);
    CHECK_EQ(type, RULE_TYPE_NETWORK);
}
