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
 * @copyright The Authors.
 * @date 2015--2019.
 * @brief Utility functions for random number handling, etc.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include "mt64.h"
#include "utils.h"

/**
 * @brief Initialises the pseudo-random number generator.
 */
void random_init()
{
    time_t now = time(0);
    unsigned char *p = (unsigned char *)&now;
    unsigned seed = 0;
    for(size_t i = 0; i < sizeof(now); i++) {
        seed = (seed * (UCHAR_MAX + 2U)) + p[i];
    }
    init_genrand64(seed);
}

/**
 * @brief Returns a uniform random float [0,1]
 * @return A random float.
 *
 * @details Mersenne Twister 64bit version.
 */
double drand()
{
    return genrand64_real1();
}

/**
 * @brief Returns a uniform random integer [min,max] not inclusive of max.
 * @return A random integer.
 */
int irand_uniform(int min, int max)
{
    return floor(rand_uniform(min, max));
}

/**
 * @brief Returns a uniform random float [min,max].
 * @return A random float.
 */
double rand_uniform(double min, double max)
{
    return min + (drand() * (max - min));
}

/**
 * @brief Returns a normal random float with specified mean and standard deviation.
 * @param mu Mean.
 * @param sigma Standard deviation.
 * @return A random float.
 *
 * @details Box-Muller transform.
 */
double rand_normal(double mu, double sigma)
{
    static const double epsilon = DBL_MIN;
    static const double two_pi = 2*M_PI;
    static double z1;
    static _Bool generate;
    generate = !generate;
    if(!generate) {
        return z1 * sigma + mu;
    }
    double u1;
    double u2;
    do {
        u1 = drand();
        u2 = drand();
    } while(u1 <= epsilon);
    double z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

/**
 * @brief Returns a float constrained within the specified range.
 * @param min Minimum value.
 * @param max Maximum value.
 * @param a The value to be constrained.
 * @return The constrained number.
 */
double constrain(double min, double max, double a)
{
    if (a < min) {return min;}
    if (a > max) {return max;}
    return a;
}

/**
 * @brief Returns an integer constrained within the specified range.
 * @param min Minimum value.
 * @param max Maximum value.
 * @param a The value to be constrained.
 * @return The constrained number.
 */
int iconstrain(int min, int max, int a)
{
    if (a < min) {return min;}
    if (a > max) {return max;}
    return a;
}

/**
 * @brief Generates a binary string from a float.
 * @param f The float to binarise.
 * @param binary The converted binary string (set by this function).
 * @param bits The number of bits to use for binarising.
 */
void float_to_binary(double f, char *binary, int bits)
{
    int a = (int)(f * pow(2, bits));
    for(int i = 0; i < bits; i++) {
        binary[i] = (a % 2) + '0';
        a /= 2;
    }
}
