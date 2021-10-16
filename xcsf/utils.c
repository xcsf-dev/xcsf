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
 * @file utils.c
 * @author Richard Preen <rpreen@gmail.com>
 * @author David Pätzel
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Utility functions for random number handling, etc.
 */

#include "utils.h"
#include <limits.h>
#include <stdbool.h>
#include <time.h>

/**
 * @brief Initialises the pseudo-random number generator.
 */
void
rand_init(void)
{
    time_t now = time(0);
    const unsigned char *p = (unsigned char *) &now;
    uint32_t seed = 0;
    for (size_t i = 0; i < sizeof(now); ++i) {
        seed = (seed * (UCHAR_MAX + 2U)) + p[i];
    }
    dsfmt_gv_init_gen_rand(seed);
}

/**
 * @brief Initialises the pseudo-random number generator with a fixed seed.
 * @param [in] seed Random number seed.
 */
void
rand_init_seed(const uint32_t seed)
{
    dsfmt_gv_init_gen_rand(seed);
}

/**
 * @brief Returns a uniform random float [min,max].
 * @param [in] min Minimum value.
 * @param [in] max Maximum value.
 * @return A random float.
 */
double
rand_uniform(const double min, const double max)
{
    return min + (dsfmt_gv_genrand_open_open() * (max - min));
}

/**
 * @brief Returns a uniform random integer [min,max] not inclusive of max.
 * @param [in] min Minimum value.
 * @param [in] max Maximum value (non-inclusive).
 * @return A random integer.
 */
int
rand_uniform_int(const int min, const int max)
{
    return (int) floor(rand_uniform(min, max));
}

/**
 * @brief Returns a random Gaussian with specified mean and standard deviation.
 * @details Box-Muller transform.
 * @param [in] mu Mean.
 * @param [in] sigma Standard deviation.
 * @return A random float.
 */
double
rand_normal(const double mu, const double sigma)
{
    static const double two_pi = 2 * M_PI;
    static double z1;
    static bool generate;
    generate = !generate;
    if (!generate) {
        return z1 * sigma + mu;
    }
    const double u1 = dsfmt_gv_genrand_open_open();
    const double u2 = dsfmt_gv_genrand_open_open();
    const double z0 = sqrt(-2 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}
