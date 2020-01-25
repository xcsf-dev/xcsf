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
 * @date 2015--2020.
 * @brief Piece-wise constant prediction functions.
 */ 

#pragma once

const double *pred_constant_compute(const XCSF *xcsf, const CL *c, const double *x);
_Bool pred_constant_crossover(const XCSF *xcsf, CL *c1, CL *c2);
_Bool pred_constant_mutate(const XCSF *xcsf, CL *c);
void pred_constant_copy(const XCSF *xcsf, CL *to, const CL *from);
void pred_constant_free(const XCSF *xcsf, const CL *c);
void pred_constant_init(const XCSF *xcsf, CL *c);
void pred_constant_print(const XCSF *xcsf, const CL *c);
void pred_constant_update(const XCSF *xcsf, const CL *c, const double *x, const double *y);
int pred_constant_size(const XCSF *xcsf, const CL *c);
size_t pred_constant_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t pred_constant_load(const XCSF *xcsf, CL *c, FILE *fp);

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
