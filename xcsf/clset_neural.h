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

double
clset_mean_cond_connections(const XCSF *xcsf, const SET *set, int layer);

double
clset_mean_cond_layers(const XCSF *xcsf, const SET *set);

double
clset_mean_cond_neurons(const XCSF *xcsf, const SET *set, int layer);

double
clset_mean_pred_connections(const XCSF *xcsf, const SET *set, int layer);

double
clset_mean_pred_eta(const XCSF *xcsf, const SET *set, int layer);

double
clset_mean_pred_layers(const XCSF *xcsf, const SET *set);

double
clset_mean_pred_neurons(const XCSF *xcsf, const SET *set, int layer);
