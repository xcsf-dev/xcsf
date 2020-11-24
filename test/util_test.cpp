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
 * @file util_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Utility tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/utils.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("UTIL")
{
    rand_init();
    // test float to binary
    char tmp[3];
    float_to_binary(1, tmp, 3);
    char correct[3] = { '1', '1', '1' };
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    correct[0] = '0';
    correct[1] = '0';
    correct[2] = '0';
    float_to_binary(0, tmp, 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    correct[0] = '0';
    correct[1] = '0';
    correct[2] = '1';
    float_to_binary(0.175, tmp, 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    correct[0] = '1';
    correct[1] = '0';
    correct[2] = '0';
    float_to_binary(0.5, tmp, 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    correct[0] = '1';
    correct[1] = '0';
    correct[2] = '1';
    float_to_binary(0.705, tmp, 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    correct[0] = '1';
    correct[1] = '0';
    float_to_binary(0.5, tmp, 2);
    for (int i = 0; i < 2; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    correct[0] = '0';
    correct[1] = '1';
    float_to_binary(0.35, tmp, 2);
    for (int i = 0; i < 2; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    correct[0] = '1';
    correct[1] = '1';
    float_to_binary(0.79, tmp, 2);
    for (int i = 0; i < 2; ++i) {
        CHECK_EQ(correct[i], tmp[i]);
    }
    // test max index
    double x[5] = { 0.214, 0.6423, 0.111, 0.775, 0.445 };
    int max = max_index(x, 5);
    CHECK_EQ(max, 3);
    x[3] = 0.1;
    max = max_index(x, 5);
    CHECK_EQ(max, 1);
    x[1] = -0.2;
    max = max_index(x, 5);
    CHECK_EQ(max, 4);
}
