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
 * @file pred_constant.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2019.
 * @brief Piece-wise constant prediction functions.
 */ 

#pragma once

double *pred_constant_compute(XCSF *xcsf, CL *c, double *x);
_Bool pred_constant_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool pred_constant_mutate(XCSF *xcsf, CL *c);
void pred_constant_copy(XCSF *xcsf, CL *to, CL *from);
void pred_constant_free(XCSF *xcsf, CL *c);
void pred_constant_init(XCSF *xcsf, CL *c);
void pred_constant_print(XCSF *xcsf, CL *c);
void pred_constant_update(XCSF *xcsf, CL *c, double *x, double *y);
int pred_constant_size(XCSF *xcsf, CL *c);
size_t pred_constant_save(XCSF *xcsf, CL *c, FILE *fp);
size_t pred_constant_load(XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Constant prediction implemented functions.
 */
static struct PredVtbl const pred_constant_vtbl = {
    &pred_constant_crossover,
    &pred_constant_mutate,
    &pred_constant_compute,
    &pred_constant_copy,
    &pred_constant_free,
    &pred_constant_init,
    &pred_constant_print,
    &pred_constant_update,
    &pred_constant_size,
    &pred_constant_save,
    &pred_constant_load
};
