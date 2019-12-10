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
 * @date 2019.
 * @brief Ternary condition functions.
 */ 

#pragma once

_Bool cond_ternary_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_ternary_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_ternary_match(XCSF *xcsf, CL *c, double *x);
_Bool cond_ternary_mutate(XCSF *xcsf, CL *c);
void cond_ternary_copy(XCSF *xcsf, CL *to, CL *from);
void cond_ternary_cover(XCSF *xcsf, CL *c, double *x);
void cond_ternary_free(XCSF *xcsf, CL *c);
void cond_ternary_init(XCSF *xcsf, CL *c);
void cond_ternary_print(XCSF *xcsf, CL *c);
void cond_ternary_update(XCSF *xcsf, CL *c, double *x, double *y);
int cond_ternary_size(XCSF *xcsf, CL *c);
size_t cond_ternary_save(XCSF *xcsf, CL *c, FILE *fp);
size_t cond_ternary_load(XCSF *xcsf, CL *c, FILE *fp);

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
