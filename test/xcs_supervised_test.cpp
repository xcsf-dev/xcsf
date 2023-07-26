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
 * @file xcs_supervised_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2023.
 * @brief High-level supervised learning function tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/pa.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcs_supervised.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("SUPERVISED")
{
    /* Test initialisation */
    const int n_samples = 5;
    const int x_dim = 4;
    const int y_dim = 1;

    double x[20] = { 0.7566081103, 0.3125093674, 0.3449376898, 0.3677518467,
                     0.7276272381, 0.2457498699, 0.2704867908, 0.0000000000,
                     0.8586376463, 0.2309959724, 0.5802303236, 0.9674486498,
                     0.5587937197, 0.6346787906, 0.0464343089, 0.4214295062,
                     0.7107445754, 0.7048862747, 0.1036188594, 0.4501471722 };

    double y[5] = { 0.1, 0.2, 0.3, 0.4, 0.5 };

    double expected[5] = { 0.361927, 0.362126, 0.314226, 0.463001, 0.413388 };

    struct XCSF xcsf;
    param_init(&xcsf, x_dim, y_dim, 1);
    rand_init_seed(2);
    xcsf_init(&xcsf);

    /* Smoke test */
    struct Input train_data;
    train_data.n_samples = n_samples;
    train_data.x_dim = x_dim;
    train_data.y_dim = y_dim;
    train_data.x = x;
    train_data.y = y;
    double *cover = (double *) calloc(y_dim, sizeof(double));
    // fit() with only train data
    xcs_supervised_fit(&xcsf, &train_data, NULL, true, 100);
    // score()
    double score = xcs_supervised_score(&xcsf, &train_data, cover);
    CHECK_EQ(doctest::Approx(score), 0.117578);
    score = xcs_supervised_score_n(&xcsf, &train_data, 10, cover);
    CHECK_EQ(doctest::Approx(score), 0.117578);
    score = xcs_supervised_score_n(&xcsf, &train_data, 2, cover);
    CHECK_EQ(doctest::Approx(score), 0.014226);
    // predict()
    double *output =
        (double *) malloc(sizeof(double) * n_samples * xcsf.pa_size);
    xcs_supervised_predict(&xcsf, x, output, n_samples, cover);
    for (int i = 0; i < n_samples; i++) {
        CHECK_EQ(doctest::Approx(output[i]), expected[i]);
    }
    // fit() with train and test data
    xcs_supervised_fit(&xcsf, &train_data, &train_data, true, 100);
    score = xcs_supervised_score(&xcsf, &train_data, cover);
    CHECK_EQ(doctest::Approx(score), 0.0367041);

    /* Test clean up */
    xcsf_free(&xcsf);
    param_free(&xcsf);
}
