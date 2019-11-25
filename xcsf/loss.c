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
 * @date 2019.
 * @brief Loss functions for calculating prediction error.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <errno.h>
#include "xcsf.h"
#include "loss.h"

double loss_mae(XCSF *xcsf, double *pred, double *y)
{
    double error = 0.0;
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        error += fabs(y[i] - pred[i]);
    }
    error /= (double)xcsf->num_y_vars;
    return error;
}
 
double loss_mse(XCSF *xcsf, double *pred, double *y)
{
    double error = 0.0;
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        error += (y[i] - pred[i]) * (y[i] - pred[i]);
    }
    error /= (double)xcsf->num_y_vars;
    return error;
}

double loss_rmse(XCSF *xcsf, double *pred, double *y)
{
    double error = loss_mse(xcsf, pred, y);
    return sqrt(error);
}

double loss_log(XCSF *xcsf, double *pred, double *y)
{
    // logistic log loss for multi-class classification
    // assumes the sum of predictions = 1 and
    // a single target y_i has a value of 1
    double error = 0.0;
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        error += y[i] * log(fmax(pred[i], 1e-15));
    }
    return -error;
}

double loss_binary_log(XCSF *xcsf, double *pred, double *y)
{
    // binary logistic log loss
    double error = 0.0;
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        error += y[i] * log(fmax(pred[i], 1e-15)) +
            (1-y[i]) * log(fmax((1-pred[i]), 1e-15));
    }
    return -error;
}
 
double loss_onehot_acc(XCSF *xcsf, double *pred, double *y)
{
    // one-hot classification error
    int p = 0;
    for(int i = 1; i < xcsf->num_y_vars; i++) {
        if(pred[i] > pred[p]) {
            p = i;
        }
    }
    if(y[p] != 1) {
        return 1;
    }
    return 0;
}
 
void loss_set_func(XCSF *xcsf)
{
    switch(xcsf->LOSS_FUNC) {
        case 0:
            xcsf->loss_ptr = &loss_mae;
            break;
        case 1:
            xcsf->loss_ptr = &loss_mse;
            break;
        case 2:
            xcsf->loss_ptr = &loss_rmse;
            break;
        case 3:
            xcsf->loss_ptr = &loss_log;
            break;
        case 4:
            xcsf->loss_ptr = &loss_binary_log;
            break;
        case 5:
            xcsf->loss_ptr = &loss_onehot_acc;
            break;
        default:
            printf("invalid loss function: %d\n", xcsf->LOSS_FUNC);
            exit(EXIT_FAILURE);
    }
}
