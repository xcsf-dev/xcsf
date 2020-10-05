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
 * @file env_csv.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief CSV input file handling functions.
 */

#pragma once

#include "env.h"
#include "xcsf.h"

/**
 * @brief CSV environment data structure.
 */
struct EnvCSV {
    struct Input *train_data;
    struct Input *test_data;
};

bool
env_csv_is_done(const struct XCSF *xcsf);

bool
env_csv_multistep(const struct XCSF *xcsf);

double
env_csv_execute(const struct XCSF *xcsf, const int action);

double
env_csv_maxpayoff(const struct XCSF *xcsf);

const double *
env_csv_get_state(const struct XCSF *xcsf);

void
env_csv_free(const struct XCSF *xcsf);

void
env_csv_init(struct XCSF *xcsf, const char *filename);

void
env_csv_reset(const struct XCSF *xcsf);

/**
 * @brief csv input environment implemented functions.
 */
static struct EnvVtbl const env_csv_vtbl = {
    &env_csv_is_done,   &env_csv_multistep, &env_csv_execute,
    &env_csv_maxpayoff, &env_csv_get_state, &env_csv_free,
    &env_csv_reset
};
