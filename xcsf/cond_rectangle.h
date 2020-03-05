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
 * @file cond_rectangle.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Hyperrectangle condition functions.
 */ 

#pragma once

/**
 * @brief Hyperrectangle condition data structure.
 */ 
typedef struct COND_RECTANGLE {
    double *center; //!< Centers
    double *spread; //!< Spreads
    double *mu; //!< Mutation rates
} COND_RECTANGLE;

_Bool cond_rectangle_crossover(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_rectangle_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_rectangle_match(const XCSF *xcsf, const CL *c, const double *x);
_Bool cond_rectangle_mutate(const XCSF *xcsf, const CL *c);
void cond_rectangle_copy(const XCSF *xcsf, CL *dest, const CL *src);
void cond_rectangle_cover(const XCSF *xcsf, const CL *c, const double *x);
void cond_rectangle_free(const XCSF *xcsf, const CL *c);
void cond_rectangle_init(const XCSF *xcsf, CL *c);
void cond_rectangle_print(const XCSF *xcsf, const CL *c);
void cond_rectangle_update(const XCSF *xcsf, const CL *c, const double *x, const double *y);
int cond_rectangle_size(const XCSF *xcsf, const CL *c);
size_t cond_rectangle_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t cond_rectangle_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Hyperrectangle condition implemented functions.
 */
static struct CondVtbl const cond_rectangle_vtbl = {
    &cond_rectangle_crossover,
    &cond_rectangle_general,
    &cond_rectangle_match,
    &cond_rectangle_mutate,
    &cond_rectangle_copy,
    &cond_rectangle_cover,
    &cond_rectangle_free,
    &cond_rectangle_init,
    &cond_rectangle_print,
    &cond_rectangle_update,
    &cond_rectangle_size,
    &cond_rectangle_save,
    &cond_rectangle_load
};      
