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
 * @file cond_rectangle_test.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Hyperrectangle condition tests.
 */ 

namespace xcsf
{ 
    TEST_SUITE_BEGIN("COND_RECTANGLE_TEST");

    TEST_CASE("COND_RECTANGLE_INIT")
    {
        xcsf.x_dim = 5;
        xcsf.y_dim = 1;
        xcsf.COND_MIN = 0;
        xcsf.COND_MAX = 1;
        xcsf.COND_SMIN = 1;
        xcsf.COND_TYPE = COND_TYPE_HYPERRECTANGLE;
        cl_init(&xcsf, &c, 1, 1);
        cond_rectangle_init(&xcsf, &c);
    }

    TEST_CASE("COND_RECTANGLE_MATCH")
    {
        const double x[5] = { 0.8455260670, 0.7566081103, 0.3125093674,
            0.3449376898, 0.3677518467}; 

        const double true_center[5] = { 0.6917788795, 0.7276272381, 0.2457498699,
            0.2704867908, 0.0000000000 };

        const double true_spread[5] = { 0.5881265924, 0.8586376463, 0.2309959724,
            0.5802303236, 0.9674486498 };

        const double false_center[5] = { 0.8992419107, 0.5587937197,
            0.6346787906, 0.0464343089, 0.4214295062 };

        const double false_spread[5] = { 0.9658827122, 0.7107445754,
            0.7048862747, 0.1036188594, 0.4501471722 };
            
        /* test for true match condition */
        COND_RECTANGLE *p = (COND_RECTANGLE *) c.cond;
        memcpy(p->center, true_center, xcsf.x_dim * sizeof(double));
        memcpy(p->spread, true_spread, xcsf.x_dim * sizeof(double));
        _Bool match = cond_rectangle_match(&xcsf, &c, x);
        REQUIRE(match == true);

        /* test for false match condition */
        memcpy(p->center, false_center, xcsf.x_dim * sizeof(double));
        memcpy(p->spread, false_spread, xcsf.x_dim * sizeof(double));
        match = cond_rectangle_match(&xcsf, &c, x);
        REQUIRE(match == false);
    }

    TEST_SUITE_END();
}
