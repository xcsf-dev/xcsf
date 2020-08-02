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
 * @date 2015--2020.
 * @brief Self-adaptive mutation functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "sam.h"

#define N_RATES (10) //!< number of mutation rates for rate selection adaptation
static const double mrates[N_RATES] = {
    0.0001, 0.001, 0.002, 0.005, 0.01, 0.01, 0.02, 0.05, 0.1, 0.5
}; //!< mutation values for rate selection adaptation

static void
sam_rate_selection_init(double *mu, int n);

static void
sam_rate_selection_adapt(double *mu, int n);

static void
sam_log_normal_init(double *mu, int n);

static void
sam_log_normal_adapt(double *mu, int n);

/**
 * @brief Initialises self-adaptive mutation rates.
 * @param xcsf The XCSF data structure.
 * @param mu The classifier's mutation rates.
 * @param n The number of mutation rates.
 */
void
sam_init(const XCSF *xcsf, double *mu, int n)
{
    switch (xcsf->SAM_TYPE) {
        case SAM_LOG_NORMAL:
            sam_log_normal_init(mu, n);
            break;
        case SAM_RATE_SELECT:
            sam_rate_selection_init(mu, n);
            break;
        default:
            printf("sam_reset(): invalid sam function: %d\n", xcsf->SAM_TYPE);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Self-adapts mutation rates.
 * @param xcsf The XCSF data structure.
 * @param mu The classifier's mutation rates.
 * @param n The number of mutation rates.
 */
void
sam_adapt(const XCSF *xcsf, double *mu, int n)
{
    switch (xcsf->SAM_TYPE) {
        case SAM_LOG_NORMAL:
            sam_log_normal_adapt(mu, n);
            break;
        case SAM_RATE_SELECT:
            sam_rate_selection_adapt(mu, n);
            break;
        default:
            printf("sam_adapt(): invalid sam function: %d\n", xcsf->SAM_TYPE);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Initialises log normal self-adaptive mutation rates.
 * @param mu The mutation rates to initialise.
 * @param n The number of mutation rates.
 */
static void
sam_log_normal_init(double *mu, int n)
{
    for (int i = 0; i < n; ++i) {
        mu[i] = rand_uniform(0, 1);
    }
}

/**
 * @brief Performs log normal self-adaptation.
 * @param mu The mutation rates to adapt.
 * @param n The number of mutation rates.
 */
static void
sam_log_normal_adapt(double *mu, int n)
{
    for (int i = 0; i < n; ++i) {
        mu[i] *= exp(rand_normal(0, 1));
        mu[i] = clamp(0.0001, 1, mu[i]);
    }
}

/**
 * @brief Initialises rate selection self-adaptive mutation.
 * @param mu The mutation rates to initialise.
 * @param n The number of mutation rates.
 */
static void
sam_rate_selection_init(double *mu, int n)
{
    for (int i = 0; i < n; ++i) {
        mu[i] = mrates[irand_uniform(0, N_RATES)];
    }
}

/**
 * @brief Performs self-adaptation via a rate selection mechanism.
 * @param mu The mutation rates to adapt.
 * @param n The number of mutation rates.
 * @details With 10% probability, randomly selects one of the possible rates.
 */
static void
sam_rate_selection_adapt(double *mu, int n)
{
    for (int i = 0; i < n; ++i) {
        if (rand_uniform(0, 1) < 0.1) {
            mu[i] = mrates[irand_uniform(0, N_RATES)];
        }
    }
}
