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

double xcs_rl_error(const XCSF *xcsf, int action,
                    double reward, _Bool reset, double max_p);
double xcs_rl_exp(XCSF *xcsf);
int xcs_rl_decision(XCSF *xcsf, const double *state);
void xcs_rl_end_step(XCSF *xcsf, const double *state,
                     int action, double reward);
void xcs_rl_end_trial(XCSF *xcsf);
void xcs_rl_init_step(XCSF *xcsf);
void xcs_rl_init_trial(XCSF *xcsf);
void xcs_rl_update(XCSF *xcsf, const double *state,
                   int action, double reward, _Bool reset);
