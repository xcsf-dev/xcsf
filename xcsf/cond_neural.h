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
 * @file cond_neural.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2019.
 * @brief Multi-layer perceptron neural network condition functions.
 */ 
 
#pragma once

_Bool cond_neural_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_neural_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cond_neural_match(XCSF *xcsf, CL *c, double *x);
_Bool cond_neural_mutate(XCSF *xcsf, CL *c);
void cond_neural_copy(XCSF *xcsf, CL *to, CL *from);
void cond_neural_cover(XCSF *xcsf, CL *c, double *x);
void cond_neural_free(XCSF *xcsf, CL *c);
void cond_neural_init(XCSF *xcsf, CL *c);
void cond_neural_print(XCSF *xcsf, CL *c);
void cond_neural_update(XCSF *xcsf, CL *c, double *x, double *y);
int cond_neural_size(XCSF *xcsf, CL *c);
size_t cond_neural_save(XCSF *xcsf, CL *c, FILE *fp);
size_t cond_neural_load(XCSF *xcsf, CL *c, FILE *fp);

static struct CondVtbl const cond_neural_vtbl = {
    &cond_neural_crossover,
    &cond_neural_general,
    &cond_neural_match,
    &cond_neural_mutate,
    &cond_neural_copy,
    &cond_neural_cover,
    &cond_neural_free,
    &cond_neural_init,
    &cond_neural_print,
    &cond_neural_update,
    &cond_neural_size,
    &cond_neural_save,
    &cond_neural_load
};      
