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
 * @file xcsf_test.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief A unit test.
 */ 

namespace xcsf
{ 
    TEST_SUITE_BEGIN("XCSF_TEST");

    TEST_CASE("CONFIG TEST") {
        config_init(&xcsf, "default.ini");
        CHECK_EQ(xcsf.ALPHA, 0.1);
    }

    TEST_SUITE_END();
}
