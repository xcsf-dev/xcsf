/*
 * Copyright (C) 2012--2019 Richard Preen <rpreen@gmail.com>
 *
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
 * @file pred_neural.h
 * @brief Multi-layer perceptron neural network prediction functions
 */ 
 
#pragma once

double *pred_neural_compute(XCSF *xcsf, CL *c, double *x);
_Bool pred_neural_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool pred_neural_mutate(XCSF *xcsf, CL *c);
void pred_neural_copy(XCSF *xcsf, CL *to,  CL *from);
void pred_neural_free(XCSF *xcsf, CL *c);
void pred_neural_init(XCSF *xcsf, CL *c);
void pred_neural_print(XCSF *xcsf, CL *c);
void pred_neural_update(XCSF *xcsf, CL *c, double *x, double *y);
int pred_neural_size(XCSF *xcsf, CL *c);
size_t pred_neural_save(XCSF *xcsf, CL *c, FILE *fp);
size_t pred_neural_load(XCSF *xcsf, CL *c, FILE *fp);
double pred_neural_eta(XCSF *xcsf, CL *c, int layer);

static struct PredVtbl const pred_neural_vtbl = {
    &pred_neural_crossover,
    &pred_neural_mutate,
    &pred_neural_compute,
    &pred_neural_copy,
    &pred_neural_free,
    &pred_neural_init,
    &pred_neural_print,
    &pred_neural_update,
    &pred_neural_size,
    &pred_neural_save,
    &pred_neural_load
};
