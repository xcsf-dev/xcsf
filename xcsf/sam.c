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
 * @file sam.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Self-adaptive mutation functions.
 */

#include "sam.h"
#include "utils.h"

#define MU_EPSILON 0.0005 //!< smallest mutation rate allowable
#define N_RATES (10) //!< number of mutation rates for rate selection adaptation

/**
 * @brief Values for rate selection adaptation.
 */
static const double mrates[N_RATES] = { 0.0005, 0.001, 0.002, 0.003, 0.005,
                                        0.01,   0.015, 0.02,  0.05,  0.1 };

/**
 * @brief Initialises self-adaptive mutation rates.
 * @param [out] mu Vector of mutation rates.
 * @param [in] N Number of mutation rates.
 * @param [in] type Vector specifying each rate type.
 */
void
sam_init(double *mu, const int N, const int *type)
{
    for (int i = 0; i < N; ++i) {
        switch (type[i]) {
            case SAM_LOG_NORMAL:
            case SAM_UNIFORM:
                mu[i] = rand_uniform(MU_EPSILON, 1);
                break;
            case SAM_RATE_SELECT:
                mu[i] = mrates[rand_uniform_int(0, N_RATES)];
                break;
            default:
                printf("sam_init(): invalid sam function: %d\n", type[i]);
                exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Self-adapts mutation rates.
 * @param [in,out] mu Vector of mutation rates.
 * @param [in] N Number of mutation rates.
 * @param [in] type Vector specifying each rate type.
 */
void
sam_adapt(double *mu, const int N, const int *type)
{
    for (int i = 0; i < N; ++i) {
        switch (type[i]) {
            case SAM_LOG_NORMAL:
                mu[i] *= exp(rand_normal(0, 1));
                mu[i] = clamp(mu[i], MU_EPSILON, 1);
                break;
            case SAM_RATE_SELECT:
                if (rand_uniform(0, 1) < 0.1) {
                    mu[i] = mrates[rand_uniform_int(0, N_RATES)];
                }
                break;
            case SAM_UNIFORM:
                if (rand_uniform(0, 1) < 0.1) {
                    mu[i] = rand_uniform(MU_EPSILON, 1);
                }
                break;
            default:
                printf("sam_adapt(): invalid sam function: %d\n", type[i]);
                exit(EXIT_FAILURE);
        }
    }
}
