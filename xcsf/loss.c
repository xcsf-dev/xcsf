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
 * @file loss.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2022.
 * @brief Loss functions for calculating prediction error.
 */

#include "loss.h"
#include "utils.h"

/**
 * @brief Mean absolute error loss function.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] pred The predicted values.
 * @param [in] y The true values.
 * @return The mean absolute error.
 */
double
loss_mae(const struct XCSF *xcsf, const double *pred, const double *y)
{
    double error = 0;
    for (int i = 0; i < xcsf->y_dim; ++i) {
        error += fabs(y[i] - pred[i]);
    }
    error /= xcsf->y_dim;
    return error;
}

/**
 * @brief Mean squared error loss function.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] pred The predicted values.
 * @param [in] y The true values.
 * @return The mean squared error.
 */
double
loss_mse(const struct XCSF *xcsf, const double *pred, const double *y)
{
    double error = 0;
    for (int i = 0; i < xcsf->y_dim; ++i) {
        error += (y[i] - pred[i]) * (y[i] - pred[i]);
    }
    error /= xcsf->y_dim;
    return error;
}

/**
 * @brief Root mean squared error loss function.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] pred The predicted values.
 * @param [in] y The true values.
 * @return The root mean squared error.
 */
double
loss_rmse(const struct XCSF *xcsf, const double *pred, const double *y)
{
    return sqrt(loss_mse(xcsf, pred, y));
}

/**
 * @brief Logistic log loss for multi-class classification.
 * @pre The sum of predictions = 1 and a single target y_i has a value of 1.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] pred The predicted values.
 * @param [in] y The true values.
 * @return The log error.
 */
double
loss_log(const struct XCSF *xcsf, const double *pred, const double *y)
{
    double error = 0;
    for (int i = 0; i < xcsf->y_dim; ++i) {
        error += y[i] * log(fmax(pred[i], DBL_EPSILON));
    }
    return -error;
}

/**
 * @brief Binary logistic log loss for binary-class classification.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] pred The predicted values.
 * @param [in] y The true values.
 * @return The log error.
 */
double
loss_binary_log(const struct XCSF *xcsf, const double *pred, const double *y)
{
    double error = 0;
    for (int i = 0; i < xcsf->y_dim; ++i) {
        error += y[i] * log(fmax(pred[i], DBL_EPSILON)) +
            (1 - y[i]) * log(fmax((1 - pred[i]), DBL_EPSILON));
    }
    return -error;
}

/**
 * @brief One-hot classification error.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] pred The predicted values.
 * @param [in] y The true values.
 * @return The one-hot classification error.
 */
double
loss_onehot(const struct XCSF *xcsf, const double *pred, const double *y)
{
    const int max_i = argmax(pred, xcsf->y_dim);
    if (y[max_i] != 1) {
        return 1;
    }
    return 0;
}

/**
 * @brief Huber loss function.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] pred The predicted values.
 * @param [in] y The true values.
 * @return The Huber loss.
 */
double
loss_huber(const struct XCSF *xcsf, const double *pred, const double *y)
{
    double error = 0;
    const double delta = xcsf->HUBER_DELTA;
    for (int i = 0; i < xcsf->y_dim; ++i) {
        const double a = y[i] - pred[i];
        if (fabs(a) > delta) {
            error += 0.5 * delta * delta + delta * (fabs(a) - delta);
        } else {
            error += 0.5 * a * a;
        }
    }
    error /= xcsf->y_dim;
    return error;
}

/**
 * @brief Sets the XCSF error function to the implemented function.
 * @param [in] xcsf The XCSF data structure.
 * @return Integer representation of the loss function.
 */
int
loss_set_func(struct XCSF *xcsf)
{
    switch (xcsf->LOSS_FUNC) {
        case LOSS_MAE:
            xcsf->loss_ptr = &loss_mae;
            return LOSS_MAE;
        case LOSS_MSE:
            xcsf->loss_ptr = &loss_mse;
            return LOSS_MSE;
        case LOSS_RMSE:
            xcsf->loss_ptr = &loss_rmse;
            return LOSS_RMSE;
        case LOSS_LOG:
            xcsf->loss_ptr = &loss_log;
            return LOSS_LOG;
        case LOSS_BINARY_LOG:
            xcsf->loss_ptr = &loss_binary_log;
            return LOSS_BINARY_LOG;
        case LOSS_ONEHOT:
            xcsf->loss_ptr = &loss_onehot;
            return LOSS_ONEHOT;
        case LOSS_HUBER:
            xcsf->loss_ptr = &loss_huber;
            return LOSS_HUBER;
        default:
            return LOSS_INVALID;
    }
}

/**
 * @brief Returns a string representation of a loss type from the integer.
 * @param [in] type Integer representation of a loss function type.
 * @return String representing the name of the loss function type.
 */
const char *
loss_type_as_string(const int type)
{
    switch (type) {
        case LOSS_MAE:
            return LOSS_STRING_MAE;
        case LOSS_MSE:
            return LOSS_STRING_MSE;
        case LOSS_RMSE:
            return LOSS_STRING_RMSE;
        case LOSS_LOG:
            return LOSS_STRING_LOG;
        case LOSS_BINARY_LOG:
            return LOSS_STRING_BINARY_LOG;
        case LOSS_ONEHOT:
            return LOSS_STRING_ONEHOT;
        case LOSS_HUBER:
            return LOSS_STRING_HUBER;
        default:
            printf("loss_type_as_string(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the integer representation of a loss type given a name.
 * @param [in] type String representation of a loss function type.
 * @return Integer representing the loss function type.
 */
int
loss_type_as_int(const char *type)
{
    if (strncmp(type, LOSS_STRING_MAE, 4) == 0) {
        return LOSS_MAE;
    }
    if (strncmp(type, LOSS_STRING_MSE, 4) == 0) {
        return LOSS_MSE;
    }
    if (strncmp(type, LOSS_STRING_RMSE, 5) == 0) {
        return LOSS_RMSE;
    }
    if (strncmp(type, LOSS_STRING_LOG, 4) == 0) {
        return LOSS_LOG;
    }
    if (strncmp(type, LOSS_STRING_BINARY_LOG, 11) == 0) {
        return LOSS_BINARY_LOG;
    }
    if (strncmp(type, LOSS_STRING_ONEHOT, 7) == 0) {
        return LOSS_ONEHOT;
    }
    if (strncmp(type, LOSS_STRING_HUBER, 6) == 0) {
        return LOSS_HUBER;
    }
    return LOSS_INVALID;
}
