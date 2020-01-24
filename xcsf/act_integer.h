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
 * @file act_integer.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief integer action functions.
 */ 

#pragma once

_Bool act_integer_crossover(const XCSF *xcsf, CL *c1, CL *c2);
_Bool act_integer_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool act_integer_mutate(const XCSF *xcsf, CL *c);
int act_integer_compute(const XCSF *xcsf, CL *c, const double *x);
void act_integer_copy(const XCSF *xcsf, CL *to, const CL *from);
void act_integer_cover(const XCSF *xcsf, CL *c, const double *x, int action);
void act_integer_free(const XCSF *xcsf, const CL *c);
void act_integer_init(const XCSF *xcsf, CL *c);
void act_integer_print(const XCSF *xcsf, const CL *c);
void act_integer_rand(const XCSF *xcsf, CL *c);
void act_integer_update(const XCSF *xcsf, CL *c, const double *x, const double *y);
size_t act_integer_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t act_integer_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Integer action implemented functions.
 */
static struct ActVtbl const act_integer_vtbl = {
    &act_integer_general,
    &act_integer_crossover,
    &act_integer_mutate,
    &act_integer_compute,
    &act_integer_copy,
    &act_integer_cover,
    &act_integer_free,
    &act_integer_init,
    &act_integer_rand,
    &act_integer_print,
    &act_integer_update,
    &act_integer_save,
    &act_integer_load
};     
