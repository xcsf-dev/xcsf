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
 * @file neural_layer_maxpool.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D maxpooling layer.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <float.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_maxpool.h"

static void malloc_layer_arrays(LAYER *l);

LAYER *neural_layer_maxpool_init(const XCSF *xcsf, int h, int w, int c, int size,
                                 int stride, int pad)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = MAXPOOL;
    l->layer_vptr = &layer_maxpool_vtbl;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->pad = pad;
    l->out_w = (w + pad - size) / stride + 1;
    l->out_h = (h + pad - size) / stride + 1;
    l->out_c = c;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = h * w * c;
    l->size = size;
    l->stride = stride;
    malloc_layer_arrays(l);
    return l;
}

LAYER *neural_layer_maxpool_copy(const XCSF *xcsf, const LAYER *src)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->height = src->height;
    l->width = src->width;
    l->channels = src->channels;
    l->pad = src->pad;
    l->out_w = src->out_w;
    l->out_h = src->out_h;
    l->out_c = src->out_c;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->n_inputs = src->n_inputs;
    l->size = src->size;
    l->stride = src->stride;
    malloc_layer_arrays(l);
    return l;
}

static void malloc_layer_arrays(LAYER *l)
{
    if(l->n_outputs < 1) {
        printf("neural_layer_maxpool: malloc() invalid size\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->indexes = calloc(l->n_outputs, sizeof(int));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
}

void neural_layer_maxpool_free(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    free(l->indexes);
    free(l->output);
    free(l->delta);
}

void neural_layer_maxpool_rand(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    (void)l;
}

void neural_layer_maxpool_forward(const XCSF *xcsf, const LAYER *l, const double *input)
{
    (void)xcsf;
    int w_offset = -l->pad / 2;
    int h_offset = w_offset;
    int h = l->out_h;
    int w = l->out_w;
    int c = l->channels;
    for(int k = 0; k < c; k++) {
        for(int i = 0; i < h; i++) {
            for(int j = 0; j < w; j++) {
                int out_index = j + w * (i + h * k);
                double max = -DBL_MAX;
                int max_i = -1;
                for(int n = 0; n < l->size; n++) {
                    for(int m = 0; m < l->size; m++) {
                        int cur_h = h_offset + i * l->stride + n;
                        int cur_w = w_offset + j * l->stride + m;
                        int index = cur_w + l->width * (cur_h + l->height * k);
                        int valid = (cur_h >= 0 && cur_h < l->height &&
                                     cur_w >= 0 && cur_w < l->width);
                        double val = (valid != 0) ? input[index] : -DBL_MAX;
                        max_i = (val > max) ? index : max_i;
                        max = (val > max) ? val : max;
                    }
                }
                l->output[out_index] = max;
                l->indexes[out_index] = max_i;
            }
        }
    }
}

void neural_layer_maxpool_backward(const XCSF *xcsf, const LAYER *l, const double *input,
                                   double *delta)
{
    (void)xcsf;
    (void)input;
    if(!delta) {
        return;
    }
    for(int i = 0; i < l->n_outputs; i++) {
        int index = l->indexes[i];
        delta[index] += l->delta[i];
    }
}

void neural_layer_maxpool_update(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    (void)l;
}

_Bool neural_layer_maxpool_mutate(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    (void)l;
    return false;
}

void neural_layer_maxpool_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void)xcsf;
    (void)l;
    (void)prev;
    printf("neural_layer_maxpool_resize(): cannot be resized\n");
    exit(EXIT_FAILURE);
}

double *neural_layer_maxpool_output(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    return l->output;
}

void neural_layer_maxpool_print(const XCSF *xcsf, const LAYER *l, _Bool print_weights)
{
    (void)xcsf;
    (void)print_weights;
    printf("maxpool in=%d, out=%d, h=%d, w=%d, c=%d, size=%d, stride=%d, pad=%d\n",
           l->n_inputs, l->n_outputs, l->height, l->width, l->channels,
           l->size, l->stride, l->pad);
}

size_t neural_layer_maxpool_save(const XCSF *xcsf, const LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&l->height, sizeof(int), 1, fp);
    s += fwrite(&l->width, sizeof(int), 1, fp);
    s += fwrite(&l->channels, sizeof(int), 1, fp);
    s += fwrite(&l->pad, sizeof(int), 1, fp);
    s += fwrite(&l->out_w, sizeof(int), 1, fp);
    s += fwrite(&l->out_h, sizeof(int), 1, fp);
    s += fwrite(&l->out_c, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->size, sizeof(int), 1, fp);
    s += fwrite(&l->stride, sizeof(int), 1, fp);
    return s;
}

size_t neural_layer_maxpool_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    layer_init(l);
    s += fread(&l->height, sizeof(int), 1, fp);
    s += fread(&l->width, sizeof(int), 1, fp);
    s += fread(&l->channels, sizeof(int), 1, fp);
    s += fread(&l->pad, sizeof(int), 1, fp);
    s += fread(&l->out_w, sizeof(int), 1, fp);
    s += fread(&l->out_h, sizeof(int), 1, fp);
    s += fread(&l->out_c, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->size, sizeof(int), 1, fp);
    s += fread(&l->stride, sizeof(int), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
