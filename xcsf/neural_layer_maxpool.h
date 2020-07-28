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
 * @file neural_layer_maxpool.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D maxpooling layer.
 */

#pragma once

LAYER *neural_layer_maxpool_init(const XCSF *xcsf, int h, int w, int c, int size,
                                 int stride, int pad);
LAYER *neural_layer_maxpool_copy(const XCSF *xcsf, const LAYER *src);
void neural_layer_maxpool_rand(const XCSF *xcsf, LAYER *l);
void neural_layer_maxpool_forward(const XCSF *xcsf, const LAYER *l, NET *net);
void neural_layer_maxpool_backward(const XCSF *xcsf, const LAYER *l, NET *net);
void neural_layer_maxpool_update(const XCSF *xcsf, const LAYER *l);
void neural_layer_maxpool_print(const XCSF *xcsf, const LAYER *l, _Bool print_weights);
_Bool neural_layer_maxpool_mutate(const XCSF *xcsf, LAYER *l);
void neural_layer_maxpool_free(const XCSF *xcsf, const LAYER *l);
double *neural_layer_maxpool_output(const XCSF *xcsf, const LAYER *l);
size_t neural_layer_maxpool_save(const XCSF *xcsf, const LAYER *l, FILE *fp);
size_t neural_layer_maxpool_load(const XCSF *xcsf, LAYER *l, FILE *fp);
void neural_layer_maxpool_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev);

static struct LayerVtbl const layer_maxpool_vtbl = {
    &neural_layer_maxpool_mutate,
    &neural_layer_maxpool_resize,
    &neural_layer_maxpool_copy,
    &neural_layer_maxpool_free,
    &neural_layer_maxpool_rand,
    &neural_layer_maxpool_print,
    &neural_layer_maxpool_update,
    &neural_layer_maxpool_backward,
    &neural_layer_maxpool_forward,
    &neural_layer_maxpool_output,
    &neural_layer_maxpool_save,
    &neural_layer_maxpool_load
};
