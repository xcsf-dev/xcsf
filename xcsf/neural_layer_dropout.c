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
 * @file neural_layer_dropout.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a dropout layer.
 */

#include "neural_layer_dropout.h"
#include "neural_activations.h"
#include "utils.h"

static void
malloc_layer_arrays(struct LAYER *l);

static void
free_layer_arrays(const struct LAYER *l);

/**
 * @brief Creates and initialises a dropout layer.
 * @param xcsf The XCSF data structure.
 * @param n_inputs The number of inputs.
 * @param probability The probability of dropping an input.
 * @return A pointer to the new layer.
 */
struct LAYER *
neural_layer_dropout_init(const struct XCSF *xcsf, int n_inputs,
                          double probability)
{
    (void) xcsf;
    struct LAYER *l = malloc(sizeof(struct LAYER));
    layer_init(l);
    l->layer_type = DROPOUT;
    l->layer_vptr = &layer_dropout_vtbl;
    l->n_inputs = n_inputs;
    l->n_outputs = n_inputs;
    l->max_outputs = n_inputs;
    l->probability = probability;
    l->scale = 1. / (1. - probability);
    malloc_layer_arrays(l);
    return l;
}

static void
malloc_layer_arrays(struct LAYER *l)
{
    if (l->n_inputs < 1 || l->n_inputs > N_INPUTS_MAX) {
        printf("neural_layer_dropout: malloc() invalid size\n");
        l->n_inputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_inputs, sizeof(double));
    l->delta = calloc(l->n_inputs, sizeof(double));
    l->state = calloc(l->n_inputs, sizeof(double));
}

static void
free_layer_arrays(const struct LAYER *l)
{
    free(l->output);
    free(l->delta);
    free(l->state);
}

void
neural_layer_dropout_free(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    free_layer_arrays(l);
}

struct LAYER *
neural_layer_dropout_copy(const struct XCSF *xcsf, const struct LAYER *src)
{
    (void) xcsf;
    struct LAYER *l = malloc(sizeof(struct LAYER));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_inputs;
    l->max_outputs = src->max_outputs;
    l->probability = src->probability;
    l->scale = src->scale;
    malloc_layer_arrays(l);
    return l;
}

void
neural_layer_dropout_rand(const struct XCSF *xcsf, struct LAYER *l)
{
    (void) xcsf;
    (void) l;
}

void
neural_layer_dropout_forward(const struct XCSF *xcsf, const struct LAYER *l,
                             const double *input)
{
    if (!xcsf->explore) {
        memcpy(l->output, input, sizeof(double) * l->n_inputs);
    } else {
        for (int i = 0; i < l->n_inputs; ++i) {
            l->state[i] = rand_uniform(0, 1);
            if (l->state[i] < l->probability) {
                l->output[i] = 0;
            } else {
                l->output[i] = input[i] * l->scale;
            }
        }
    }
}

void
neural_layer_dropout_backward(const struct XCSF *xcsf, const struct LAYER *l,
                              const double *input, double *delta)
{
    (void) xcsf;
    (void) input;
    if (!delta) {
        return;
    }
    for (int i = 0; i < l->n_inputs; ++i) {
        if (l->state[i] < l->probability) {
            delta[i] = 0;
        } else {
            delta[i] += l->delta[i] * l->scale;
        }
    }
}

void
neural_layer_dropout_update(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    (void) l;
}

_Bool
neural_layer_dropout_mutate(const struct XCSF *xcsf, struct LAYER *l)
{
    (void) xcsf;
    (void) l;
    return false;
}

void
neural_layer_dropout_resize(const struct XCSF *xcsf, struct LAYER *l,
                            const struct LAYER *prev)
{
    (void) xcsf;
    l->n_inputs = prev->n_outputs;
    l->n_outputs = prev->n_outputs;
    l->max_outputs = prev->n_outputs;
    free_layer_arrays(l);
    malloc_layer_arrays(l);
}

double *
neural_layer_dropout_output(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    return l->output;
}

void
neural_layer_dropout_print(const struct XCSF *xcsf, const struct LAYER *l,
                           _Bool print_weights)
{
    (void) xcsf;
    (void) print_weights;
    printf("dropout in = %d, out = %d prob = %f\n", l->n_inputs, l->n_outputs,
           l->probability);
}

size_t
neural_layer_dropout_save(const struct XCSF *xcsf, const struct LAYER *l,
                          FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->probability, sizeof(double), 1, fp);
    s += fwrite(&l->scale, sizeof(double), 1, fp);
    return s;
}

size_t
neural_layer_dropout_load(const struct XCSF *xcsf, struct LAYER *l, FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    layer_init(l);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->probability, sizeof(double), 1, fp);
    s += fread(&l->scale, sizeof(double), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
