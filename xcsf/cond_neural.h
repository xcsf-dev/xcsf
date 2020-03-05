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
 * @date 2016--2020.
 * @brief Multi-layer perceptron neural network condition functions.
 */ 
 
#pragma once

/**
 * @brief Multi-layer perceptron neural network condition data structure.
 */ 
typedef struct COND_NEURAL {
    NET net; //!< Neural network
} COND_NEURAL;

_Bool cond_neural_crossover(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_neural_general(const XCSF *xcsf, const CL *c1, const CL *c2);
_Bool cond_neural_match(const XCSF *xcsf, const CL *c, const double *x);
_Bool cond_neural_mutate(const XCSF *xcsf, const CL *c);
void cond_neural_copy(const XCSF *xcsf, CL *dest, const CL *src);
void cond_neural_cover(const XCSF *xcsf, const CL *c, const double *x);
void cond_neural_free(const XCSF *xcsf, const CL *c);
void cond_neural_init(const XCSF *xcsf, CL *c);
void cond_neural_print(const XCSF *xcsf, const CL *c);
void cond_neural_update(const XCSF *xcsf, const CL *c, const double *x, const double *y);
int cond_neural_size(const XCSF *xcsf, const CL *c);
size_t cond_neural_save(const XCSF *xcsf, const CL *c, FILE *fp);
size_t cond_neural_load(const XCSF *xcsf, CL *c, FILE *fp);

/**
 * @brief Multi-layer perceptron neural network condition implemented functions.
 */
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
