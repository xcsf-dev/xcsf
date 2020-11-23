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
 * @file cond_ternary_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Ternary condition tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/cond_ternary.h"
#include "../xcsf/condition.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("COND_TERNARY")
{
    struct XCSF xcsf;
    struct Cl c1;
    rand_init();
    param_init(&xcsf, 5, 1, 1);
    cond_param_set_type(&xcsf, COND_TYPE_TERNARY);
    cond_param_set_bits(&xcsf, 2);
    cl_init(&xcsf, &c1, 1, 1);
    cond_ternary_init(&xcsf, &c1);
    struct CondTernary *p = (struct CondTernary *) c1.cond;
    CHECK_EQ(p->length, 10);
    const double x[5] = { 0.8455260670, 0.0566081103, 0.3125093674,
                          0.3449376898, 0.5677518467 };
    /* test for true match condition */
    const char *true_1 = "1100010110";
    memcpy(p->string, true_1, sizeof(char) * 10);
    bool match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, true);
    const char *true_2 = "1#00#101#0";
    memcpy(p->string, true_2, sizeof(char) * 10);
    match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, true);
    /* test for false match condition */
    const char *false_1 = "1100000110";
    memcpy(p->string, false_1, sizeof(char) * 10);
    match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, false);
    const char *false_2 = "0#00#101#0";
    memcpy(p->string, false_2, sizeof(char) * 10);
    match = cond_ternary_match(&xcsf, &c1, x);
    CHECK_EQ(match, false);
    /* test general */
    struct Cl c2;
    cl_init(&xcsf, &c2, 1, 1);
    cond_ternary_init(&xcsf, &c2);
    struct CondTernary *p2 = (struct CondTernary *) c2.cond;
    const char *spec = "0000#101#0";
    memcpy(p2->string, spec, sizeof(char) * 10);
    bool general = cond_ternary_general(&xcsf, &c1, &c2);
    CHECK_EQ(general, true);
    general = cond_ternary_general(&xcsf, &c2, &c1);
    CHECK_EQ(general, false);
}
