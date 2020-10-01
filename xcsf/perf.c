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
 * @file perf.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief System performance printing.
 */

#include "perf.h"

/**
 * @brief Displays the current training and test performance.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] error The current training error.
 * @param [in] terror The current testing error.
 * @param [in] trial The number of learning trials executed.
 */
void
perf_print(const struct XCSF *xcsf, double *error, double *terror,
           const int trial)
{
    if (trial % xcsf->PERF_TRIALS == 0 && trial > 0) {
        *error /= xcsf->PERF_TRIALS;
        *terror /= xcsf->PERF_TRIALS;
        printf("%d %.5f %.5f %d\n", trial, *error, *terror, xcsf->pset.size);
        fflush(stdout);
        *error = 0;
        *terror = 0;
    }
}
