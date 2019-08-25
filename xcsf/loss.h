 /*
 * Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
 *
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
 * 
 */

double loss_mae(XCSF *xcsf, double *pred, double *y);
double loss_mse(XCSF *xcsf, double *pred, double *y);
double loss_rmse(XCSF *xcsf, double *pred, double *y);
double loss_log(XCSF *xcsf, double *pred, double *y);
double loss_binary_log(XCSF *xcsf, double *pred, double *y);
double loss_onehot_acc(XCSF *xcsf, double *pred, double *y);
void loss_set_func(XCSF *xcsf);
