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
 * @file neural_layer_connected_test.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Fully-connected neural network layer tests.
 */ 

namespace xcsf
{ 
    TEST_SUITE_BEGIN("NEURAL_LAYER_CONNECTED TEST");

    LAYER *l;

    TEST_CASE("NEURAL_LAYER_CONNECTED_INIT")
    {
        xcsf.x_dim = 10;
        xcsf.y_dim = 2;
        xcsf.PRED_TYPE = 5;
        xcsf.PRED_ETA = 0.1;
        uint32_t o = 0;
        o |= LAYER_EVOLVE_ETA;
        o |= LAYER_SGD_WEIGHTS;
        o |= LAYER_EVOLVE_WEIGHTS;
        l = neural_layer_connected_init(&xcsf, 10, 2, 2, LOGISTIC, o);
        CHECK_EQ(l->n_weights, 20);
        CHECK_EQ(l->n_outputs, 2);
    }

    TEST_CASE("NEURAL_LAYER_CONNECTED_FORWARD")
    {
        const double x[10] = { -0.4792173279, -0.2056298252, -0.1775459629,
            -0.0814486626, 0.0923277094, 0.2779675621, -0.3109822596,
            -0.6788371120, -0.0714929928, -0.1332985280 };

        const double y[2] = { 0.7936726123, 0.0963342482 };

        const double orig_weights[20] = { 0.3326639519, -0.4446678553,
            0.1033557369, -1.2581317787, 2.8042169798, 0.2236021733,
            -1.2206964138, -0.2022042865, -1.5489524535, -2.0932767781,
            5.4797621223, 0.3326639519, -0.4446678553, 0.1033557369,
            -1.2581317787, 2.8042169798, 0.2236021733, -1.2206964138,
            -0.2022042865, -1.5489524535 };

        const double orig_biases[2] = { 0.1033557369, -1.2581317787 };

        memcpy(l->weights, orig_weights, 20 * sizeof(double));
        memcpy(l->biases, orig_biases, 2 * sizeof(double));
        neural_layer_connected_forward(&xcsf, l, x);
        double output_error = 0;
        for(int i = 0; i < 2; i++) {
            output_error += fabs(l->output[i] - y[i]);
        }
        REQUIRE(doctest::Approx(output_error) == 0);
    }

    TEST_SUITE_END();
}
