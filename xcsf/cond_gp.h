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
 * @file cond_gp.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Tree GP condition functions.
 */ 

#pragma once

_Bool cond_gp_crossover(const XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_gp_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_gp_match(const XCSF *xcsf, CL *c, const double *x);
_Bool cond_gp_mutate(const XCSF *xcsf, CL *c);
void cond_gp_copy(const XCSF *xcsf, CL *to, const CL *from);
void cond_gp_cover(const XCSF *xcsf, CL *c, const double *x);
void cond_gp_free(const XCSF *xcsf, CL *c);
void cond_gp_init(const XCSF *xcsf, CL *c);
void cond_gp_print(const XCSF *xcsf, const CL *c);
void cond_gp_update(const XCSF *xcsf, CL *c, const double *x, const double *y);
int cond_gp_size(const XCSF *xcsf, const CL *c);
size_t cond_gp_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t cond_gp_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Tree GP condition implemented functions.
 */
static struct CondVtbl const cond_gp_vtbl = {
    &cond_gp_crossover,
    &cond_gp_general,
    &cond_gp_match,
    &cond_gp_mutate,
    &cond_gp_copy,
    &cond_gp_cover,
    &cond_gp_free,
    &cond_gp_init,
    &cond_gp_print,
    &cond_gp_update,
    &cond_gp_size,
    &cond_gp_save,
    &cond_gp_load
};      
