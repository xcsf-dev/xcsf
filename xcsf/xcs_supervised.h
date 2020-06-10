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
 * @file xcs_supervised.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Supervised regression learning functions.
 */

#pragma once

double xcs_supervised_fit(XCSF *xcsf, const INPUT *train_data,
                          const INPUT *test_data, _Bool shuffle);
double xcs_supervised_score(XCSF *xcsf, const INPUT *test_data);
void xcs_supervised_predict(XCSF *xcsf, const double *x, double *pred, int n_samples);
