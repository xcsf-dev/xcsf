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

#include "xcsf.h"

#define LOSS_MAE (0) //!< Mean absolute error
#define LOSS_MSE (1) //!< Mean squared error
#define LOSS_RMSE (2) //!< Root mean squared error
#define LOSS_LOG (3) //!< Log loss
#define LOSS_BINARY_LOG (4) //!< Binary log loss
#define LOSS_ONEHOT (5) //!< One-hot encoding classification error
#define LOSS_HUBER (6) //!< Huber loss
#define LOSS_NUM (7) //!< Total number of selectable loss functions

#define LOSS_STRING_MAE ("mae\0") //!< Mean absolute error
#define LOSS_STRING_MSE ("mse\0") //!< Mean squared error
#define LOSS_STRING_RMSE ("rmse\0") //!< Root mean squared error
#define LOSS_STRING_LOG ("log\0") //!< Log loss
#define LOSS_STRING_BINARY_LOG ("binary_log\0") //!< Binary log loss
#define LOSS_STRING_ONEHOT ("onehot\0") //!< One-hot classification error
#define LOSS_STRING_HUBER ("huber\0") //!< Huber loss

double
loss_huber(const struct XCSF *xcsf, const double *pred, const double *y);

double
loss_mae(const struct XCSF *xcsf, const double *pred, const double *y);

double
loss_mse(const struct XCSF *xcsf, const double *pred, const double *y);

double
loss_rmse(const struct XCSF *xcsf, const double *pred, const double *y);

double
loss_log(const struct XCSF *xcsf, const double *pred, const double *y);

double
loss_binary_log(const struct XCSF *xcsf, const double *pred, const double *y);

double
loss_onehot(const struct XCSF *xcsf, const double *pred, const double *y);

void
loss_set_func(struct XCSF *xcsf);

const char *
loss_type_as_string(const int type);

int
loss_type_as_int(const char *type);
