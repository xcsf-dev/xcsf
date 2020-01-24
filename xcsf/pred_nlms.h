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
 * @file pred_nlms.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Normalised least mean squares prediction functions.
 */ 
 
#pragma once

const double *pred_nlms_compute(const XCSF *xcsf, CL *c, const double *x);
_Bool pred_nlms_crossover(const XCSF *xcsf, CL *c1, CL *c2);
_Bool pred_nlms_mutate(const XCSF *xcsf, CL *c);
void pred_nlms_copy(const XCSF *xcsf, CL *to, const CL *from);
void pred_nlms_free(const XCSF *xcsf, CL *c);
void pred_nlms_init(const XCSF *xcsf, CL *c);
void pred_nlms_print(const XCSF *xcsf, const CL *c);
void pred_nlms_update(const XCSF *xcsf, CL *c, const double *x, const double *y);
int pred_nlms_size(const XCSF *xcsf, const CL *c);
size_t pred_nlms_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t pred_nlms_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Normalised least mean squares prediction implemented functions.
 */
static struct PredVtbl const pred_nlms_vtbl = {
    &pred_nlms_crossover,
    &pred_nlms_mutate,
    &pred_nlms_compute,
    &pred_nlms_copy,
    &pred_nlms_free,
    &pred_nlms_init,
    &pred_nlms_print,
    &pred_nlms_update,
    &pred_nlms_size,
    &pred_nlms_save,
    &pred_nlms_load
};
