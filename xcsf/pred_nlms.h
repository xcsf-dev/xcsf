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
 * @date 2015--2019.
 * @brief Normalised least mean squares prediction functions.
 */ 
 
#pragma once

double *pred_nlms_compute(XCSF *xcsf, CL *c, double *x);
_Bool pred_nlms_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool pred_nlms_mutate(XCSF *xcsf, CL *c);
void pred_nlms_copy(XCSF *xcsf, CL *to, CL *from);
void pred_nlms_free(XCSF *xcsf, CL *c);
void pred_nlms_init(XCSF *xcsf, CL *c);
void pred_nlms_print(XCSF *xcsf, CL *c);
void pred_nlms_update(XCSF *xcsf, CL *c, double *x, double *y);
int pred_nlms_size(XCSF *xcsf, CL *c);
size_t pred_nlms_save(XCSF *xcsf, CL *c, FILE *fp);
size_t pred_nlms_load(XCSF *xcsf, CL *c, FILE *fp);

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
