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
 * @copyright The Authors.
 * @date 2015--2019.
 * @brief Utility functions for random number handling, etc.
 */

#pragma once

double rand_normal(double mu, double sigma);
double rand_uniform(double min, double max);
int irand_uniform(int min, int max);
void random_init();

/**
 * @brief Returns a float clamped within the specified range.
 * @param min Minimum value.
 * @param max Maximum value.
 * @param a The value to be clamped.
 * @return The clamped number.
 */
static inline double
clamp(double min, double max, double a)
{
    return (a < min) ? min : (a > max) ? max : a;
}

/**
 * @brief Returns an integer clamped within the specified range.
 * @param min Minimum value.
 * @param max Maximum value.
 * @param a The value to be clamped.
 * @return The clamped number.
 */
static inline int
iclamp(int min, int max, int a)
{
    return (a < min) ? min : (a > max) ? max : a;
}
