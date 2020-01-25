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
 * @file cond_dummy.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019-2020.
 * @brief Always-matching dummy condition functions.
 */ 

#pragma once

_Bool cond_dummy_crossover(const XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_dummy_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_dummy_match(const XCSF *xcsf, const CL *c, const double *x);
_Bool cond_dummy_mutate(const XCSF *xcsf, const CL *c);
void cond_dummy_copy(const XCSF *xcsf, CL *to, const CL *from);
void cond_dummy_cover(const XCSF *xcsf, const CL *c, const double *x);
void cond_dummy_free(const XCSF *xcsf, const CL *c);
void cond_dummy_init(const XCSF *xcsf, CL *c);
void cond_dummy_print(const XCSF *xcsf, const CL *c);
void cond_dummy_update(const XCSF *xcsf, const CL *c, const double *x, const double *y);
int cond_dummy_size(const XCSF *xcsf, const CL *c);
size_t cond_dummy_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t cond_dummy_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Dummy condition implemented functions.
 */
static struct CondVtbl const cond_dummy_vtbl = {
    &cond_dummy_crossover,
    &cond_dummy_general,
    &cond_dummy_match,
    &cond_dummy_mutate,
    &cond_dummy_copy,
    &cond_dummy_cover,
    &cond_dummy_free,
    &cond_dummy_init,
    &cond_dummy_print,
    &cond_dummy_update,
    &cond_dummy_size,
    &cond_dummy_save,
    &cond_dummy_load
};
