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
 * @file neural_layer_avgpool.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of an average pooling layer.
 */

#include "neural_layer_avgpool.h"
#include "neural_activations.h"
#include "utils.h"
#include "xcsf.h"

static void
malloc_layer_arrays(struct LAYER *l)
{
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX) {
        printf("neural_layer_avgpool: malloc() invalid size\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
}

/**
 * @brief Creates and initialises an average pooling layer.
 * @param xcsf The XCSF data structure.
 * @param h The input height.
 * @param w The input width.
 * @param c The number of input channels.
 * @return A pointer to the new layer.
 */
struct LAYER *
neural_layer_avgpool_init(const struct XCSF *xcsf, int h, int w, int c)
{
    (void) xcsf;
    struct LAYER *l = malloc(sizeof(struct LAYER));
    layer_init(l);
    l->layer_type = AVGPOOL;
    l->layer_vptr = &layer_avgpool_vtbl;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->out_w = 1;
    l->out_h = 1;
    l->out_c = c;
    l->n_outputs = l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = h * w * c;
    malloc_layer_arrays(l);
    return l;
}

struct LAYER *
neural_layer_avgpool_copy(const struct XCSF *xcsf, const struct LAYER *src)
{
    (void) xcsf;
    if (src->layer_type != AVGPOOL) {
        printf("neural_layer_avgpool_copy(): incorrect source layer type\n");
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
    malloc_layer_arrays(l);
    return l;
}

void
neural_layer_avgpool_free(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    free(l->output);
    free(l->delta);
}

void
neural_layer_avgpool_rand(const struct XCSF *xcsf, struct LAYER *l)
{
    (void) xcsf;
    (void) l;
}

void
neural_layer_avgpool_forward(const struct XCSF *xcsf, const struct LAYER *l,
                             const double *input)
{
    (void) xcsf;
    int n = l->height * l->width;
    for (int k = 0; k < l->channels; ++k) {
        l->output[k] = 0;
        for (int i = 0; i < n; ++i) {
            l->output[k] += input[i + n * k];
        }
        l->output[k] /= n;
    }
}

void
neural_layer_avgpool_backward(const struct XCSF *xcsf, const struct LAYER *l,
                              const double *input, double *delta)
{
    (void) xcsf;
    (void) input;
    if (!delta) {
        return;
    }
    int n = l->height * l->width;
    for (int k = 0; k < l->channels; ++k) {
        for (int i = 0; i < n; ++i) {
            delta[i + n * k] += l->delta[k] / n;
        }
    }
}

void
neural_layer_avgpool_update(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    (void) l;
}

_Bool
neural_layer_avgpool_mutate(const struct XCSF *xcsf, struct LAYER *l)
{
    (void) xcsf;
    (void) l;
    return false;
}

void
neural_layer_avgpool_resize(const struct XCSF *xcsf, struct LAYER *l,
                            const struct LAYER *prev)
{
    (void) xcsf;
    int h = prev->out_h;
    int w = prev->out_w;
    int c = prev->out_c;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->out_c = c;
    l->n_outputs = l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = h * w * c;
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
}

double *
neural_layer_avgpool_output(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    return l->output;
}

void
neural_layer_avgpool_print(const struct XCSF *xcsf, const struct LAYER *l,
                           _Bool print_weights)
{
    (void) xcsf;
    (void) print_weights;
    printf("avgpool in=%d, out=%d, h=%d, w=%d, c=%d\n", l->n_inputs,
           l->n_outputs, l->height, l->width, l->channels);
}

size_t
neural_layer_avgpool_save(const struct XCSF *xcsf, const struct LAYER *l,
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
    return s;
}

size_t
neural_layer_avgpool_load(const struct XCSF *xcsf, struct LAYER *l, FILE *fp)
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
    malloc_layer_arrays(l);
    return s;
}
