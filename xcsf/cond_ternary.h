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
 * @file cond_ternary.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Ternary condition functions.
 */ 

#pragma once

_Bool cond_ternary_crossover(const XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_ternary_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_ternary_match(const XCSF *xcsf, CL *c, const double *x);
_Bool cond_ternary_mutate(const XCSF *xcsf, const CL *c);
void cond_ternary_copy(const XCSF *xcsf, CL *to, const CL *from);
void cond_ternary_cover(const XCSF *xcsf, CL *c, const double *x);
void cond_ternary_free(const XCSF *xcsf, const CL *c);
void cond_ternary_init(const XCSF *xcsf, CL *c);
void cond_ternary_print(const XCSF *xcsf, const CL *c);
void cond_ternary_update(const XCSF *xcsf, CL *c, const double *x, const double *y);
int cond_ternary_size(const XCSF *xcsf, const CL *c);
size_t cond_ternary_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t cond_ternary_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Ternary condition implemented functions.
 */
static struct CondVtbl const cond_ternary_vtbl = {
    &cond_ternary_crossover,
    &cond_ternary_general,
    &cond_ternary_match,
    &cond_ternary_mutate,
    &cond_ternary_copy,
    &cond_ternary_cover,
    &cond_ternary_free,
    &cond_ternary_init,
    &cond_ternary_print,
    &cond_ternary_update,
    &cond_ternary_size,
    &cond_ternary_save,
    &cond_ternary_load
};
