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
 * @file utils.h
 * @author Richard Preen <rpreen@gmail.com>
 * @author David Pätzel
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Utility functions for random number handling, etc.
 */

#pragma once

#include "../lib/cJSON/cJSON.h"
#include "../lib/dSFMT/dSFMT.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double
rand_normal(const double mu, const double sigma);

double
rand_uniform(const double min, const double max);

int
rand_uniform_int(const int min, const int max);

void
rand_init(void);

void
rand_init_seed(const uint32_t seed);

/**
 * @brief Returns a float clamped within the specified range.
 * @param [in] a The value to be clamped.
 * @param [in] min Minimum value.
 * @param [in] max Maximum value.
 * @return The clamped number.
 */
static inline double
clamp(const double a, const double min, const double max)
{
    return (a < min) ? min : (a > max) ? max : a;
}

/**
 * @brief Returns an integer clamped within the specified range.
 * @param [in] a The value to be clamped.
 * @param [in] min Minimum value.
 * @param [in] max Maximum value.
 * @return The clamped number.
 */
static inline int
clamp_int(const int a, const int min, const int max)
{
    return (a < min) ? min : (a > max) ? max : a;
}

/**
 * @brief Returns the index of the largest element in vector X.
 * @param [in] X Vector with N elements.
 * @param [in] N The number of elements in vector X.
 * @return The index of the largest element.
 */
static inline int
max_index(const double *X, const int N)
{
    if (N < 1) {
        printf("max_index() error: N < 1\n");
        exit(EXIT_FAILURE);
    }
    int max_i = 0;
    double max = X[0];
    for (int i = 1; i < N; ++i) {
        if (X[i] > max) {
            max_i = i;
            max = X[i];
        }
    }
    return max_i;
}

/**
 * @brief Generates a binary string from a float.
 * @param [in] f The float to binarise.
 * @param [out] binary The converted binary string.
 * @param [in] bits The number of bits to use for binarising.
 */
static inline void
float_to_binary(const double f, char *binary, const int bits)
{
    if (f >= 1) {
        for (int i = 0; i < bits; ++i) {
            binary[i] = '1';
        }
    } else if (f <= 0) {
        for (int i = 0; i < bits; ++i) {
            binary[i] = '0';
        }
    } else {
        int a = (int) (f * pow(2, bits));
        for (int i = 0; i < bits; ++i) {
            binary[bits - 1 - i] = (a % 2) + '0';
            a /= 2;
        }
    }
}
