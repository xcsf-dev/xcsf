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
 * @file xcs_rl.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Reinforcement learning functions.
 */

#pragma once

#include "xcsf.h"

double
xcs_rl_error(struct XCSF *xcsf, const int action, const double reward,
             const bool reset, const double max_p);

double
xcs_rl_exp(struct XCSF *xcsf);

int
xcs_rl_decision(struct XCSF *xcsf, const double *state);

void
xcs_rl_end_step(struct XCSF *xcsf, const double *state, const int action,
                const double reward);

void
xcs_rl_end_trial(struct XCSF *xcsf);

void
xcs_rl_init_step(struct XCSF *xcsf);

void
xcs_rl_init_trial(struct XCSF *xcsf);

void
xcs_rl_update(struct XCSF *xcsf, const double *state, const int action,
              const double reward, const bool reset);

double
xcs_rl_fit(struct XCSF *xcsf, const double *state, const int action,
           const double reward);
