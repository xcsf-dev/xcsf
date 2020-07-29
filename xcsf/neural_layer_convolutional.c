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
 * @file neural_layer_convolutional.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D convolutional layer.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include "xcsf.h"
#include "utils.h"
#include "blas.h"
#include "sam.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_convolutional.h"

#define N_MU 4 //!< Number of mutation rates applied to a convolutional layer

static int convolutional_out_height(const LAYER *l);
static int convolutional_out_width(const LAYER *l);
static double im2col_get_pixel(const double *im, int height, int width, int row, int col,
                               int channel, int pad);
static void im2col(const double *data_im, int channels, int height, int width,
                   int ksize, int stride, int pad, double *data_col);
static void col2im_add_pixel(double *im, int height, int width, int row, int col,
                             int channel, int pad, double val);
static void col2im(const double *data_col, int channels, int height, int width,
                   int ksize, int stride, int pad, double *data_im);

LAYER *neural_layer_convolutional_init(const XCSF *xcsf, int h, int w, int c,
                                       int n_filters, int kernel_size, int stride,
                                       int pad, int f, uint32_t o)
{
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = CONVOLUTIONAL;
    l->layer_vptr = &layer_convolutional_vtbl;
    l->options = o;
    l->function = f;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->n_filters = n_filters;
    l->stride = stride;
    l->size = kernel_size;
    l->pad = pad;
    l->n_weights = l->channels * n_filters * kernel_size * kernel_size;
    l->weights = malloc(l->n_weights * sizeof(double));
    l->weight_updates = calloc(l->n_weights, sizeof(double));
    l->weight_active = malloc(l->n_weights * sizeof(_Bool));
    l->n_active = l->n_weights;
    for(int i = 0; i < l->n_weights; i++) {
        l->weights[i] = rand_normal(0, 0.1);
        l->weight_active[i] = true;
    }
    l->out_h = convolutional_out_height(l);
    l->out_w = convolutional_out_width(l);
    l->out_c = n_filters;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->n_inputs = l->width * l->height * l->channels;
    l->max_outputs = l->n_outputs;
    l->biases = calloc(l->n_filters, sizeof(double));
    l->bias_updates = calloc(l->n_filters, sizeof(double));
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->workspace_size = l->out_h * l->out_w * l->size * l->size * c * sizeof(double);
    l->temp = malloc(l->workspace_size);
    layer_init_eta(xcsf, l);
    l->mu = malloc(N_MU * sizeof(double));
    sam_init(xcsf, l->mu, N_MU);
    return l;
}

static int convolutional_out_height(const LAYER *l)
{
    return (l->height + 2 * l->pad - l->size) / l->stride + 1;
}

static int convolutional_out_width(const LAYER *l)
{
    return (l->width + 2 * l->pad - l->size) / l->stride + 1;
}

LAYER *neural_layer_convolutional_copy(const XCSF *xcsf, const LAYER *src)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->options = src->options;
    l->function = src->function;
    l->height = src->height;
    l->width = src->width;
    l->channels = src->channels;
    l->n_filters = src->n_filters;
    l->stride = src->stride;
    l->size = src->size;
    l->pad = src->pad;
    l->n_weights = src->n_weights;
    l->weights = malloc(src->n_weights * sizeof(double));
    memcpy(l->weights, src->weights, src->n_weights * sizeof(double));
    l->weight_updates = calloc(src->n_weights, sizeof(double));
    l->weight_active = malloc(src->n_weights * sizeof(_Bool));
    memcpy(l->weight_active, src->weight_active, src->n_weights * sizeof(_Bool));
    l->n_active = src->n_active;
    l->out_h = src->out_h;
    l->out_w = src->out_w;
    l->out_c = src->out_c;
    l->n_outputs = src->n_outputs;
    l->n_inputs = src->n_inputs;
    l->max_outputs = src->max_outputs;
    l->state = calloc(src->n_outputs, sizeof(double));
    l->output = calloc(src->n_outputs, sizeof(double));
    l->delta = calloc(src->n_outputs, sizeof(double));
    l->bias_updates = calloc(src->n_filters, sizeof(double));
    l->biases = malloc(src->n_filters * sizeof(double));
    memcpy(l->biases, src->biases, src->n_filters * sizeof(double));
    l->workspace_size = src->workspace_size;
    l->temp = malloc(src->workspace_size);
    l->eta = src->eta;
    l->mu = malloc(N_MU * sizeof(double));
    memcpy(l->mu, src->mu, N_MU * sizeof(double));
    return l;
}

void neural_layer_convolutional_free(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    free(l->state);
    free(l->output);
    free(l->weights);
    free(l->biases);
    free(l->bias_updates);
    free(l->weight_updates);
    free(l->delta);
    free(l->weight_active);
    free(l->temp);
    free(l->mu);
}

void neural_layer_convolutional_rand(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    l->n_active = l->n_weights;
    for(int i = 0; i < l->n_weights; i++) {
        l->weights[i] = rand_normal(0, 1);
        l->weight_active[i] = true;
    }
    for(int i = 0; i < l->n_filters; i++) {
        l->biases[i] = rand_normal(0, 1);
    }
}

void neural_layer_convolutional_forward(const XCSF *xcsf, const LAYER *l,
                                        const double *input)
{
    (void)xcsf;
    int m = l->n_filters;
    int k = l->size * l->size * l->channels;
    int n = l->out_w * l->out_h;
    const double *a = l->weights;
    double *b = l->temp;
    double *c = l->state;
    memset(l->state, 0, l->n_outputs * sizeof(double));
    if(l->size == 1) {
        blas_gemm(0, 0, m, n, k, 1, a, k, input, n, 1, c, n);
    } else {
        im2col(input, l->channels, l->height, l->width, l->size, l->stride, l->pad, b);
        blas_gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
    for(int i = 0; i < l->n_filters; i++) {
        for(int j = 0; j < n; j++) {
            l->state[i * n + j] += l->biases[i];
        }
    }
    neural_activate_array(l->state, l->output, l->n_outputs, l->function);
}

void neural_layer_convolutional_backward(const XCSF *xcsf, const LAYER *l,
        const double *input, double *delta)
{
    (void)xcsf;
    int m = l->n_filters;
    int n = l->size * l->size * l->channels;
    int k = l->out_w * l->out_h;
    if(l->options & LAYER_SGD_WEIGHTS) {
        neural_gradient_array(l->state, l->delta, l->n_outputs, l->function);
        for(int i = 0; i < l->n_filters; i++) {
            l->bias_updates[i] += blas_sum(l->delta + k * i, k);
        }
        const double *a = l->delta;
        double *b = l->temp;
        double *c = l->weight_updates;
        if(l->size == 1) {
            blas_gemm(0, 1, m, n, k, 1, a, k, input, k, 1, c, n);
        } else {
            im2col(input, l->channels, l->height, l->width, l->size, l->stride, l->pad, b);
            blas_gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
        }
    }
    if(delta) {
        const double *a = l->weights;
        const double *b = l->delta;
        double *c = l->temp;
        if(l->size == 1) {
            c = delta;
        }
        blas_gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
        if(l->size != 1) {
            col2im(l->temp, l->channels, l->height, l->width, l->size, l->stride, l->pad, delta);
        }
    }
}

void neural_layer_convolutional_update(const XCSF *xcsf, const LAYER *l)
{
    if(l->options & LAYER_SGD_WEIGHTS) {
        blas_axpy(l->n_filters, l->eta, l->bias_updates, 1, l->biases, 1);
        blas_axpy(l->n_weights, l->eta, l->weight_updates, 1, l->weights, 1);
        blas_scal(l->n_filters, xcsf->PRED_MOMENTUM, l->bias_updates, 1);
        blas_scal(l->n_weights, xcsf->PRED_MOMENTUM, l->weight_updates, 1);
        layer_weight_clamp(l);
    }
}

void neural_layer_convolutional_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void)xcsf;
    (void)l;
    (void)prev;
    printf("neural_layer_convolutional_resize(): cannot be resized\n");
    exit(EXIT_FAILURE);
}

_Bool neural_layer_convolutional_mutate(const XCSF *xcsf, LAYER *l)
{
    sam_adapt(xcsf, l->mu, N_MU);
    _Bool mod = false;
    if((l->options & LAYER_EVOLVE_ETA) && layer_mutate_eta(xcsf, l, l->mu[0])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_CONNECT) && layer_mutate_connectivity(l, l->mu[1])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_WEIGHTS) && layer_mutate_weights(l, l->mu[2])) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_FUNCTIONS) && layer_mutate_functions(l, l->mu[3])) {
        mod = true;
    }
    return mod;
}

double *neural_layer_convolutional_output(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    return l->output;
}

void neural_layer_convolutional_print(const XCSF *xcsf, const LAYER *l,
                                      _Bool print_weights)
{
    (void)xcsf;
    printf("convolutional %s, in=%d, out=%d, filters=%d, size=%d, stride=%d, pad=%d",
           neural_activation_string(l->function), l->n_inputs, l->n_outputs,
           l->size, l->n_filters, l->stride, l->pad);
    printf("\n");
    (void)print_weights;
}

size_t neural_layer_convolutional_save(const XCSF *xcsf, const LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->height, sizeof(int), 1, fp);
    s += fwrite(&l->width, sizeof(int), 1, fp);
    s += fwrite(&l->channels, sizeof(int), 1, fp);
    s += fwrite(&l->n_filters, sizeof(int), 1, fp);
    s += fwrite(&l->stride, sizeof(int), 1, fp);
    s += fwrite(&l->size, sizeof(int), 1, fp);
    s += fwrite(&l->pad, sizeof(int), 1, fp);
    s += fwrite(&l->out_h, sizeof(int), 1, fp);
    s += fwrite(&l->out_w, sizeof(int), 1, fp);
    s += fwrite(&l->out_c, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_weights, sizeof(int), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(&l->workspace_size, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(l->weights, sizeof(double), l->n_weights, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fwrite(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->n_filters, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->n_filters, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t neural_layer_convolutional_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->height, sizeof(int), 1, fp);
    s += fread(&l->width, sizeof(int), 1, fp);
    s += fread(&l->channels, sizeof(int), 1, fp);
    s += fread(&l->n_filters, sizeof(int), 1, fp);
    s += fread(&l->stride, sizeof(int), 1, fp);
    s += fread(&l->size, sizeof(int), 1, fp);
    s += fread(&l->pad, sizeof(int), 1, fp);
    s += fread(&l->out_h, sizeof(int), 1, fp);
    s += fread(&l->out_w, sizeof(int), 1, fp);
    s += fread(&l->out_c, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_weights, sizeof(int), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    s += fread(&l->workspace_size, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    if(l->n_inputs < 1 || l->n_outputs < 1 || l->n_filters < 1 || l->n_weights < 1) {
        printf("neural_layer_convolutional_load(): read error\n");
        l->n_outputs = 1;
        l->n_weights = 1;
        l->n_filters = 1;
        exit(EXIT_FAILURE);
    }
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->temp = malloc(l->workspace_size);
    l->weights = malloc(l->n_weights * sizeof(double));
    l->weight_updates = malloc(l->n_weights * sizeof(double));
    l->weight_active = malloc(l->n_weights * sizeof(_Bool));
    l->biases = malloc(l->n_filters * sizeof(double));
    l->bias_updates = malloc(l->n_filters * sizeof(double));
    l->mu = malloc(N_MU * sizeof(double));
    s += fread(l->weights, sizeof(double), l->n_weights, fp);
    s += fread(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fread(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fread(l->biases, sizeof(double), l->n_filters, fp);
    s += fread(l->bias_updates, sizeof(double), l->n_filters, fp);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    return s;
}

static double im2col_get_pixel(const double *im, int height, int width, int row, int col,
                               int channel, int pad)
{
    row -= pad;
    col -= pad;
    if(row < 0 || col < 0 || row >= height || col >= width) {
        return 0;
    }
    return im[col + width * (row + height * channel)];
}

static void im2col(const double *data_im, int channels, int height, int width,
                   int ksize, int stride, int pad, double *data_col)
{
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for(int c = 0; c < channels_col; c++) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for(int h = 0; h < height_col; h++) {
            for(int w = 0; w < width_col; w++) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}

static void col2im_add_pixel(double *im, int height, int width, int row, int col,
                             int channel, int pad, double val)
{
    row -= pad;
    col -= pad;
    if(row < 0 || col < 0 || row >= height || col >= width) {
        return;
    }
    im[col + width * (row + height * channel)] += val;
}

static void col2im(const double *data_col, int channels, int height, int width,
                   int ksize, int stride, int pad, double *data_im)
{
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for(int c = 0; c < channels_col; c++) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for(int h = 0; h < height_col; h++) {
            for(int w = 0; w < width_col; w++) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, im_row, im_col, c_im, pad, val);
            }
        }
    }
}
