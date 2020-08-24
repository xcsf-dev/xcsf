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
 * @file neural_layer_upsample.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D upsampling layer.
 */

#include "neural_layer_upsample.h"
#include "neural_activations.h"
#include "utils.h"
#include "xcsf.h"

static void
malloc_layer_arrays(struct LAYER *l)
{
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX) {
        printf("neural_layer_upsample: malloc() invalid size\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
}

/**
 * @brief Creates and initialises a 2D upsampling layer.
 * @param xcsf The XCSF data structure.
 * @param h The input height.
 * @param w The input width.
 * @param c The number of input channels.
 * @param stride The strides of the upsampling operation.
 * @return A pointer to the new layer.
 */
struct LAYER *
neural_layer_upsample_init(const struct XCSF *xcsf, const int h, const int w,
                           const int c, const int stride)
{
    (void) xcsf;
    struct LAYER *l = malloc(sizeof(struct LAYER));
    layer_init(l);
    l->layer_type = UPSAMPLE;
    l->layer_vptr = &layer_upsample_vtbl;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->out_c = c;
    l->stride = stride;
    l->out_w = w * l->stride;
    l->out_h = h * l->stride;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = h * w * c;
    malloc_layer_arrays(l);
    return l;
}

struct LAYER *
neural_layer_upsample_copy(const struct XCSF *xcsf, const struct LAYER *src)
{
    (void) xcsf;
    if (src->layer_type != UPSAMPLE) {
        printf("neural_layer_upsample_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct LAYER *l = malloc(sizeof(struct LAYER));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->height = src->height;
    l->width = src->width;
    l->channels = src->channels;
    l->out_w = src->out_w;
    l->out_h = src->out_h;
    l->out_c = src->out_c;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->n_inputs = src->n_inputs;
    l->stride = src->stride;
    malloc_layer_arrays(l);
    return l;
}

void
neural_layer_upsample_free(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    free(l->output);
    free(l->delta);
}

void
neural_layer_upsample_rand(const struct XCSF *xcsf, struct LAYER *l)
{
    (void) xcsf;
    (void) l;
}

void
neural_layer_upsample_forward(const struct XCSF *xcsf, const struct LAYER *l,
                              const double *input)
{
    (void) xcsf;
    const int w = l->width;
    const int h = l->height;
    const int c = l->channels;
    const int s = l->stride;
    for (int k = 0; k < c; ++k) {
        for (int j = 0; j < h * s; ++j) {
            for (int i = 0; i < w * s; ++i) {
                const int in_index = k * w * h + (j / s) * w + i / s;
                const int out_index = k * w * h * s * s + j * w * s + i;
                l->output[out_index] = input[in_index];
            }
        }
    }
}

void
neural_layer_upsample_backward(const struct XCSF *xcsf, const struct LAYER *l,
                               const double *input, double *delta)
{
    (void) xcsf;
    (void) input;
    if (delta) {
        const int w = l->width;
        const int h = l->height;
        const int c = l->channels;
        const int s = l->stride;
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < h * s; ++j) {
                for (int i = 0; i < w * s; ++i) {
                    const int in_index = k * w * h + (j / s) * w + i / s;
                    const int out_index = k * w * h * s * s + j * w * s + i;
                    delta[in_index] += l->delta[out_index];
                }
            }
        }
    }
}

void
neural_layer_upsample_update(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    (void) l;
}

_Bool
neural_layer_upsample_mutate(const struct XCSF *xcsf, struct LAYER *l)
{
    (void) xcsf;
    (void) l;
    return false;
}

void
neural_layer_upsample_resize(const struct XCSF *xcsf, struct LAYER *l,
                             const struct LAYER *prev)
{
    (void) xcsf;
    l->width = prev->out_w;
    l->height = prev->out_h;
    l->channels = prev->out_c;
    l->out_c = prev->out_c;
    l->out_w = l->width * l->stride;
    l->out_h = l->height * l->stride;
    l->n_inputs = l->height * l->width * l->channels;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
}

double *
neural_layer_upsample_output(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    return l->output;
}

void
neural_layer_upsample_print(const struct XCSF *xcsf, const struct LAYER *l,
                            const _Bool print_weights)
{
    (void) xcsf;
    (void) print_weights;
    printf("upsample in=%d, out=%d, h=%d, w=%d, c=%d, stride=%d\n", l->n_inputs,
           l->n_outputs, l->height, l->width, l->channels, l->stride);
}

size_t
neural_layer_upsample_save(const struct XCSF *xcsf, const struct LAYER *l,
                           FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fwrite(&l->height, sizeof(int), 1, fp);
    s += fwrite(&l->width, sizeof(int), 1, fp);
    s += fwrite(&l->channels, sizeof(int), 1, fp);
    s += fwrite(&l->out_w, sizeof(int), 1, fp);
    s += fwrite(&l->out_h, sizeof(int), 1, fp);
    s += fwrite(&l->out_c, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->stride, sizeof(int), 1, fp);
    return s;
}

size_t
neural_layer_upsample_load(const struct XCSF *xcsf, struct LAYER *l, FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fread(&l->height, sizeof(int), 1, fp);
    s += fread(&l->width, sizeof(int), 1, fp);
    s += fread(&l->channels, sizeof(int), 1, fp);
    s += fread(&l->out_w, sizeof(int), 1, fp);
    s += fread(&l->out_h, sizeof(int), 1, fp);
    s += fread(&l->out_c, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->stride, sizeof(int), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
