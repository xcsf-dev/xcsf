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
 * @file serialization_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief Serialization tests.
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

TEST_CASE("Serialization")
{
    struct XCSF xcsf;
    param_init(&xcsf, 4, 1, 1);
    param_set_random_state(&xcsf, 1);
    xcsf_init(&xcsf);

    /* test serialisation */
    size_t s = xcsf_save(&xcsf, "temp.bin");
    size_t r = xcsf_load(&xcsf, "temp.bin");
    CHECK_EQ(s, r);

    /* test param export and import */
    char *json_str = param_json_export(&xcsf);
    param_json_import(&xcsf, json_str);
    free(json_str);

    /* test print pset*/
    CAPTURE(xcsf_print_pset(&xcsf, true, true, true));

    /* test print params */
    CAPTURE(param_print(&xcsf));

    /* test store */
    xcsf_store_pset(&xcsf);

    /* test retrieve */
    xcsf_retrieve_pset(&xcsf);

    /* test clean up */
    xcsf_free(&xcsf);
    param_free(&xcsf);
}
