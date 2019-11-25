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
 * @date 2019.
 * @brief integer action functions.
 */ 

#pragma once

_Bool act_integer_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool act_integer_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool act_integer_mutate(XCSF *xcsf, CL *c);
int act_integer_compute(XCSF *xcsf, CL *c, double *x);
void act_integer_copy(XCSF *xcsf, CL *to, CL *from);
void act_integer_cover(XCSF *xcsf, CL *c, int action);
void act_integer_free(XCSF *xcsf, CL *c);
void act_integer_init(XCSF *xcsf, CL *c);
void act_integer_print(XCSF *xcsf, CL *c);
void act_integer_rand(XCSF *xcsf, CL *c);
void act_integer_update(XCSF *xcsf, CL *c, double *x, double *y);
size_t act_integer_save(XCSF *xcsf, CL *c, FILE *fp);
size_t act_integer_load(XCSF *xcsf, CL *c, FILE *fp);

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
