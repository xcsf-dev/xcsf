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
 * @file cond_dgp.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Dynamical GP graph condition functions.
 */ 

#pragma once

_Bool cond_dgp_crossover(const XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_dgp_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_dgp_match(const XCSF *xcsf, const CL *c, const double *x);
_Bool cond_dgp_mutate(const XCSF *xcsf, const CL *c);
void cond_dgp_copy(const XCSF *xcsf, CL *to, const CL *from);
void cond_dgp_cover(const XCSF *xcsf, const CL *c, const double *x);
void cond_dgp_free(const XCSF *xcsf, const CL *c);
void cond_dgp_init(const XCSF *xcsf, CL *c);
void cond_dgp_print(const XCSF *xcsf, const CL *c);
void cond_dgp_update(const XCSF *xcsf, const CL *c, const double *x, const double *y);
int cond_dgp_size(const XCSF *xcsf, const CL *c);
size_t cond_dgp_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t cond_dgp_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Dynamical GP graph condition implemented functions.
 */
static struct CondVtbl const cond_dgp_vtbl = {
    &cond_dgp_crossover,
    &cond_dgp_general,
    &cond_dgp_match,
    &cond_dgp_mutate,
    &cond_dgp_copy,
    &cond_dgp_cover,
    &cond_dgp_free,
    &cond_dgp_init,
    &cond_dgp_print,
    &cond_dgp_update,
    &cond_dgp_size,
    &cond_dgp_save,
    &cond_dgp_load
};
