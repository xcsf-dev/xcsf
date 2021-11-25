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
 * @file sam.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Self-adaptive mutation functions.
 */

#pragma once

#include "xcsf.h"

#define SAM_LOG_NORMAL (0) //!< Log normal self-adaptation
#define SAM_RATE_SELECT (1) //!< Ten normally distributed rates
#define SAM_UNIFORM (2) //!< Uniformly random self-adaptation

void
sam_adapt(double *mu, const int N, const int *type);

void
sam_init(double *mu, const int N, const int *type);
