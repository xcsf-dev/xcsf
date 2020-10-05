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
 * @file env_mux.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief The real multiplexer problem environment.
 */

#pragma once

#include "env.h"
#include "xcsf.h"

/**
 * @brief Real multiplexer environment data structure.
 */
struct EnvMux {
    double *state; //!< Current state
    int pos_bits; //!< Number of position bits
};

bool
env_mux_is_done(const struct XCSF *xcsf);

bool
env_mux_multistep(const struct XCSF *xcsf);

double
env_mux_execute(const struct XCSF *xcsf, const int action);

double
env_mux_maxpayoff(const struct XCSF *xcsf);

const double *
env_mux_get_state(const struct XCSF *xcsf);

void
env_mux_free(const struct XCSF *xcsf);

void
env_mux_init(struct XCSF *xcsf, int bits);

void
env_mux_reset(const struct XCSF *xcsf);

/**
 * @brief Real multiplexer environment implemented functions.
 */
static struct EnvVtbl const env_mux_vtbl = {
    &env_mux_is_done,   &env_mux_multistep, &env_mux_execute,
    &env_mux_maxpayoff, &env_mux_get_state, &env_mux_free,
    &env_mux_reset
};
