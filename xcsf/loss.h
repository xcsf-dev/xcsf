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
 * @file loss.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Loss functions for calculating prediction error.
 */ 

#pragma once

#define LOSS_MAE 0 //!< Mean absolute error
#define LOSS_MSE 1 //!< Mean squared error
#define LOSS_RMSE 2 //!< Root mean squared error
#define LOSS_LOG 3 //!< Log loss
#define LOSS_BINARY_LOG 4 //!< Binary log loss
#define LOSS_ONEHOT_ACC 5 //!< One-hot encoding classification error
#define LOSS_NUM 6 //!< Total number of selectable loss functions

double loss_mae(const XCSF *xcsf, const double *pred, const double *y);
double loss_mse(const XCSF *xcsf, const double *pred, const double *y);
double loss_rmse(const XCSF *xcsf, const double *pred, const double *y);
double loss_log(const XCSF *xcsf, const double *pred, const double *y);
double loss_binary_log(const XCSF *xcsf, const double *pred, const double *y);
double loss_onehot_acc(const XCSF *xcsf, const double *pred, const double *y);
void loss_set_func(XCSF *xcsf);
