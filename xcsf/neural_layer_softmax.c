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
 * @file neural_layer_softmax.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a softmax layer.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_softmax.h"

static void
free_layer_arrays(const LAYER *l);

static void
malloc_layer_arrays(LAYER *l);

/**
 * @brief Creates and initialises a softmax layer.
 * @param xcsf The XCSF data structure.
 * @param n_inputs The number of inputs.
 * @param temperature The scaling of the logits.
 * @return A pointer to the new layer.
 */
LAYER *
neural_layer_softmax_init(const XCSF *xcsf, int n_inputs, double temperature)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = SOFTMAX;
    l->layer_vptr = &layer_softmax_vtbl;
    l->scale = temperature;
    l->n_inputs = n_inputs;
    l->n_outputs = n_inputs;
    l->max_outputs = n_inputs;
    malloc_layer_arrays(l);
    return l;
}

static void
malloc_layer_arrays(LAYER *l)
{
    if (l->n_inputs < 1 || l->n_inputs > N_INPUTS_MAX) {
        printf("neural_layer_softmax: malloc() invalid size\n");
        l->n_inputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_inputs, sizeof(double));
    l->delta = calloc(l->n_inputs, sizeof(double));
}

static void
free_layer_arrays(const LAYER *l)
{
    free(l->output);
    free(l->delta);
}

LAYER *
neural_layer_softmax_copy(const XCSF *xcsf, const LAYER *src)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->scale = src->scale;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    malloc_layer_arrays(l);
    return l;
}

void
neural_layer_softmax_rand(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    (void)l;
}

void
neural_layer_softmax_forward(const XCSF *xcsf, const LAYER *l,
                             const double *input)
{
    (void)xcsf;
    double largest = input[0];
    for (int i = 1; i < l->n_inputs; ++i) {
        if (input[i] > largest) {
            largest = input[i];
        }
    }
    double sum = 0;
    for (int i = 0; i < l->n_inputs; ++i) {
        double e = exp((input[i] / l->scale) - (largest / l->scale));
        sum += e;
        l->output[i] = e;
    }
    for (int i = 0; i < l->n_inputs; ++i) {
        l->output[i] /= sum;
    }
}

void
neural_layer_softmax_backward(const XCSF *xcsf, const LAYER *l,
                              const double *input,
                              double *delta)
{
    (void)xcsf;
    (void)input;
    for (int i = 0; i < l->n_inputs; ++i) {
        delta[i] += l->delta[i];
    }
}

void
neural_layer_softmax_update(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    (void)l;
}

void
neural_layer_softmax_print(const XCSF *xcsf, const LAYER *l,
                           _Bool print_weights)
{
    (void)xcsf;
    (void)print_weights;
    printf("softmax in = %d, out = %d, temp = %f\n",
           l->n_inputs, l->n_outputs, l->scale);
}

_Bool
neural_layer_softmax_mutate(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    (void)l;
    return false;
}

void
neural_layer_softmax_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void)xcsf;
    l->n_inputs = prev->n_outputs;
    l->n_outputs = prev->n_outputs;
    l->max_outputs = prev->n_outputs;
    free_layer_arrays(l);
    malloc_layer_arrays(l);
}

void
neural_layer_softmax_free(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    free_layer_arrays(l);
}

double *
neural_layer_softmax_output(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    return l->output;
}

size_t
neural_layer_softmax_save(const XCSF *xcsf, const LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->scale, sizeof(double), 1, fp);
    return s;
}

size_t
neural_layer_softmax_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    layer_init(l);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->scale, sizeof(double), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
