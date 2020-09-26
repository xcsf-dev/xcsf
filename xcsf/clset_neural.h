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
 * @file clset_neural.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Functions operating on sets of neural classifiers.
 */

#include "xcsf.h"

double
clset_mean_cond_connections(const struct XCSF *xcsf, const struct Set *set,
                            const int layer);

double
clset_mean_cond_layers(const struct XCSF *xcsf, const struct Set *set);

double
clset_mean_cond_neurons(const struct XCSF *xcsf, const struct Set *set,
                        const int layer);

double
clset_mean_pred_connections(const struct XCSF *xcsf, const struct Set *set,
                            const int layer);

double
clset_mean_pred_eta(const struct XCSF *xcsf, const struct Set *set,
                    const int layer);

double
clset_mean_pred_layers(const struct XCSF *xcsf, const struct Set *set);

double
clset_mean_pred_neurons(const struct XCSF *xcsf, const struct Set *set,
                        const int layer);
