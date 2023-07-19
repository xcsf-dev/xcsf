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
 * @date 2020--2023.
 * @brief Recursive least mean squares unit tests.
 */

#include "../lib/doctest/doctest/doctest.h"

extern "C" {
#include "../xcsf/cl.h"
#include "../xcsf/param.h"
#include "../xcsf/pred_rls.h"
#include "../xcsf/prediction.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

TEST_CASE("PRED_RLS")
{
    /* test initialisation */
    struct XCSF xcsf;
    struct Cl c;
    rand_init();
    param_init(&xcsf, 10, 1, 1);
    pred_param_set_type(&xcsf, PRED_TYPE_RLS_LINEAR);
    pred_param_set_x0(&xcsf, 1);
    pred_param_set_scale_factor(&xcsf, 1000);
    pred_param_set_lambda(&xcsf, 1);
    cl_init(&xcsf, &c, 1, 1);
    pred_rls_init(&xcsf, &c);
    struct PredRLS *p = (struct PredRLS *) c.pred;
    CHECK_EQ(p->n, 11);
    CHECK_EQ(p->n_weights, 11);

    /* test one forward pass of input */
    const double x[10] = { -0.4792173279, -0.2056298252, -0.1775459629,
                           -0.0814486626, 0.0923277094,  0.2779675621,
                           -0.3109822596, -0.6788371120, -0.0714929928,
                           -0.1332985280 };
    const double orig_weights[11] = {
        0.3326639519,  -0.4446678553, 0.1033557369,  -1.2581317787,
        2.8042169798,  0.2236021733,  -1.2206964138, -0.2022042865,
        -1.5489524535, -2.0932767781, 5.4797621223
    };
    memcpy(p->weights, orig_weights, sizeof(double) * 11);
    pred_rls_compute(&xcsf, &c, x);
    CHECK_EQ(doctest::Approx(c.prediction[0]), 0.7343893899);

    /* test one backward pass of input */
    const double y[1] = { -0.8289711363 };
    const double orig_matrix[121] = {
        359.2527520933,  365.5458205478,  60.4529914784,   -43.9947052728,
        -13.5841371391,  109.7982766671,  115.3644158136,  -31.0528908351,
        220.6431759838,  -87.3080507605,  88.1241585637,   365.5458205478,
        701.6622252028,  -207.5401364638, 34.6047881875,   -39.9348946644,
        -71.7410651849,  -99.7839013172,  59.7013343643,   82.1749298766,
        62.4796315063,   2.5785225955,    60.4529914784,   -207.5401364638,
        660.4451979152,  14.5938249650,   -99.9815372193,  -21.4480835504,
        -84.8701956147,  75.1408338287,   379.0968152519,  33.2722467513,
        91.8072038700,   -43.9947052728,  34.6047881875,   14.5938249650,
        816.1486807003,  -209.6896853432, 155.7799317097,  -183.0261843905,
        -205.2877220046, -30.7160080608,  6.7357934831,    -39.2139859587,
        -13.5841371391,  -39.9348946644,  -99.9815372193,  -209.6896853432,
        720.1824326889,  172.6026502229,  -246.8984543081, -214.6226348053,
        86.8163701245,   21.6085251537,   -17.2300656790,  109.7982766671,
        -71.7410651849,  -21.4480835504,  155.7799317097,  172.6026502229,
        859.6410324332,  136.1207770026,  172.3225576178,  2.8038851343,
        4.7203774022,    22.7457751762,   115.3644158136,  -99.7839013172,
        -84.8701956147,  -183.0261843905, -246.8984543081, 136.1207770026,
        755.9118069104,  -193.5019068277, 13.1863818100,   35.7363200832,
        -38.7889670725,  -31.0528908351,  59.7013343643,   75.1408338287,
        -205.2877220046, -214.6226348053, 172.3225576178,  -193.5019068277,
        759.2824245631,  -112.9318121612, 5.3940936253,    -64.2781320876,
        220.6431759838,  82.1749298766,   379.0968152519,  -30.7160080608,
        86.8163701245,   2.8038851343,    13.1863818100,   -112.9318121612,
        438.8075106331,  2.5731527976,    -158.0653964682, -87.3080507605,
        62.4796315063,   33.2722467513,   6.7357934831,    21.6085251537,
        4.7203774022,    35.7363200832,   5.3940936253,    2.5731527976,
        985.2143907713,  7.6541515765,    88.1241585637,   2.5785225955,
        91.8072038700,   -39.2139859587,  -17.2300656790,  22.7457751762,
        -38.7889670725,  -64.2781320876,  -158.0653964682, 7.6541515765,
        948.0191060269
    };
    const double new_weights[11] = {
        0.0946131867,  -0.3075330137, 1.0021313882,  -0.8308518983,
        2.9259524607,  -0.2862486966, -2.7938207202, 0.4469019892,
        -1.1284685743, -1.5021336612, 5.2697089817
    };
    const double new_matrix[121] = {
        347.5068080680,  372.3123527739,  104.8005426593,  -22.9117839267,
        -7.5774429599,   84.6411225856,   37.7430310010,   0.9754287363,
        241.3907678047,  -58.1397598809,  77.7596784082,   372.3123527739,
        697.7642023390,  -233.0876048981, 22.4594666703,   -43.3951946325,
        -57.2486843729,  -55.0682445218,  41.2506541923,   70.2227829594,
        45.6765393245,   8.5492294904,    104.8005426593,  -233.0876048981,
        493.0082263515,  -65.0060719380,  -122.6601892685, 73.5343355551,
        208.1942297143,  -45.7841031855,  300.7629760834,  -76.8544718726,
        130.9389512348,  -22.9117839267,  22.4594666703,   -65.0060719380,
        778.3067182724,  -220.4711653721, 200.9347783232,  -43.7027253943,
        -262.7756975062, -67.9560839150,  -45.6186857389,  -20.6106689805,
        -7.5774429599,   -43.3951946325,  -122.6601892685, -220.4711653721,
        717.1107021508,  185.4676298840,  -207.2040781619, -231.0014222679,
        76.2063724305,   6.6923121473,    -11.9298307651,  84.6411225856,
        -57.2486843729,  73.5343355551,   200.9347783232,  185.4676298840,
        805.7601012691,  -30.1266641184,  240.9199703668,  47.2405317875,
        67.1922558969,   0.5474040887,    37.7430310010,   -55.0682445218,
        208.1942297143,  -43.7027253943,  -207.2040781619, -30.1266641184,
        242.9620289054,  18.1526449899,   150.2938637798,  228.4907828847,
        -107.2811462280, 0.9754287363,    41.2506541923,   -45.7841031855,
        -262.7756975062, -231.0014222679, 240.9199703668,  18.1526449899,
        671.9490222965,  -169.5054259758, -74.1407091652,  -36.0167269681,
        241.3907678047,  70.2227829594,   300.7629760834,  -67.9560839150,
        76.2063724305,   47.2405317875,   150.2938637798,  -169.5054259758,
        402.1597481851,  -48.9486143741,  -139.7579702419, -58.1397598809,
        45.6765393245,   -76.8544718726,  -45.6186857389,  6.6923121473,
        67.1922558969,   228.4907828847,  -74.1407091652,  -48.9486143741,
        912.7817969326,  33.3919016747,   77.7596784082,   8.5492294904,
        130.9389512348,  -20.6106689805,  -11.9298307651,  0.5474040887,
        -107.2811462280, -36.0167269681,  -139.7579702419, 33.3919016747,
        938.8736130242
    };
    memcpy(p->matrix, orig_matrix, sizeof(double) * 121);
    pred_rls_update(&xcsf, &c, x, y);
    double weight_error = 0;
    for (int i = 0; i < 11; ++i) {
        weight_error += fabs(p->weights[i] - new_weights[i]);
    }
    CHECK_EQ(doctest::Approx(weight_error), 0);
    double matrix_error = 0;
    for (int i = 0; i < 121; ++i) {
        matrix_error += fabs(p->matrix[i] - new_matrix[i]);
    }
    CHECK_EQ(doctest::Approx(matrix_error), 0);

    /* test convergence on one input */
    for (int i = 0; i < 200; ++i) {
        pred_rls_compute(&xcsf, &c, x);
        pred_rls_update(&xcsf, &c, x, y);
    }
    pred_rls_compute(&xcsf, &c, x);
    CHECK_EQ(doctest::Approx(c.prediction[0]), y[0]);

    /* test copy */
    struct Cl dest_cl;
    cl_init(&xcsf, &dest_cl, 1, 1);
    pred_rls_copy(&xcsf, &dest_cl, &c);
    struct PredRLS *dest_pred = (struct PredRLS *) dest_cl.pred;
    struct PredRLS *src_pred = (struct PredRLS *) c.pred;
    CHECK_EQ(dest_pred->n, src_pred->n);
    CHECK_EQ(dest_pred->n_weights, src_pred->n_weights);
    CHECK(check_array_eq(dest_pred->weights, src_pred->weights,
                         src_pred->n_weights));

    /* test print */
    CAPTURE(pred_rls_print(&xcsf, &c));

    /* test crossover */
    CHECK(!pred_rls_crossover(&xcsf, &c, &dest_cl));

    /* test mutation */
    CHECK(!pred_rls_mutate(&xcsf, &c));

    /* test size */
    CHECK_EQ(pred_rls_size(&xcsf, &c), src_pred->n_weights);

    /* test import and export */
    char *json_str = pred_rls_json_export(&xcsf, &c);
    struct Cl new_cl;
    cl_init(&xcsf, &new_cl, 1, 1);
    pred_rls_init(&xcsf, &new_cl);
    cJSON *json = cJSON_Parse(json_str);
    pred_rls_json_import(&xcsf, &new_cl, json);
    struct PredRLS *new_pred = (struct PredRLS *) new_cl.pred;
    CHECK_EQ(new_pred->n, src_pred->n);
    CHECK_EQ(new_pred->n_weights, src_pred->n_weights);
    CHECK(check_array_eq(new_pred->weights, src_pred->weights,
                         src_pred->n_weights));
    free(json_str);

    /* test save */
    FILE *fp = fopen("temp.bin", "wb");
    size_t s = pred_rls_save(&xcsf, &c, fp);
    fclose(fp);

    /* test load */
    fp = fopen("temp.bin", "rb");
    size_t r = pred_rls_load(&xcsf, &c, fp);
    CHECK_EQ(s, r);

    /* parameter export */
    json_str = pred_rls_param_json_export(&xcsf);

    /* parameter import */
    json = cJSON_Parse(json_str);
    char *json_rtn = pred_rls_param_json_import(&xcsf, json->child);
    CHECK(json_rtn == NULL);

    /* clean up */
    free(json_str);
    free(json_rtn);
    pred_rls_free(&xcsf, &new_cl);
    param_free(&xcsf);
}
