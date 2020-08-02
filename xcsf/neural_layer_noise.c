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
 * @file neural_layer_noise.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a Gaussian noise adding layer.
 */

#include "neural_layer_noise.h"
#include "neural_activations.h"
#include "utils.h"

static void
free_layer_arrays(const LAYER *l);

static void
malloc_layer_arrays(LAYER *l);

/**
 * @brief Creates and initialises a Gaussian noise layer.
 * @param xcsf The XCSF data structure.
 * @param n_inputs The number of inputs.
 * @param probability The probability of adding noise to an input.
 * @param std The standard deviation of the Gaussian noise added.
 * @return A pointer to the new layer.
 */
LAYER *
neural_layer_noise_init(const struct XCSF *xcsf, int n_inputs,
                        double probability, double std)
{
    (void) xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = NOISE;
    l->layer_vptr = &layer_noise_vtbl;
    l->n_inputs = n_inputs;
    l->n_outputs = n_inputs;
    l->max_outputs = n_inputs;
    l->probability = probability;
    l->scale = std;
    malloc_layer_arrays(l);
    return l;
}

static void
malloc_layer_arrays(LAYER *l)
{
    if (l->n_inputs < 1 || l->n_inputs > N_INPUTS_MAX) {
        printf("neural_layer_noise: malloc() invalid size\n");
        l->n_inputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_inputs, sizeof(double));
    l->delta = calloc(l->n_inputs, sizeof(double));
    l->state = calloc(l->n_inputs, sizeof(double));
}

static void
free_layer_arrays(const LAYER *l)
{
    free(l->output);
    free(l->delta);
    free(l->state);
}

LAYER *
neural_layer_noise_copy(const struct XCSF *xcsf, const LAYER *src)
{
    (void) xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->probability = src->probability;
    l->scale = src->scale;
    malloc_layer_arrays(l);
    return l;
}

void
neural_layer_noise_free(const struct XCSF *xcsf, const LAYER *l)
{
    (void) xcsf;
    free_layer_arrays(l);
}

void
neural_layer_noise_rand(const struct XCSF *xcsf, LAYER *l)
{
    (void) xcsf;
    (void) l;
}

void
neural_layer_noise_forward(const struct XCSF *xcsf, const LAYER *l,
                           const double *input)
{
    if (!xcsf->explore) {
        for (int i = 0; i < l->n_inputs; ++i) {
            l->output[i] = input[i];
        }
    } else {
        for (int i = 0; i < l->n_inputs; ++i) {
            l->state[i] = rand_uniform(0, 1);
            if (l->state[i] < l->probability) {
                l->output[i] = input[i] + rand_normal(0, l->scale);
            } else {
                l->output[i] = input[i];
            }
        }
    }
}

void
neural_layer_noise_backward(const struct XCSF *xcsf, const LAYER *l,
                            const double *input, double *delta)
{
    (void) xcsf;
    (void) input;
    if (!delta) {
        return;
    }
    for (int i = 0; i < l->n_inputs; ++i) {
        delta[i] += l->delta[i];
    }
}

void
neural_layer_noise_update(const struct XCSF *xcsf, const LAYER *l)
{
    (void) xcsf;
    (void) l;
}

_Bool
neural_layer_noise_mutate(const struct XCSF *xcsf, LAYER *l)
{
    (void) xcsf;
    (void) l;
    return false;
}

void
neural_layer_noise_resize(const struct XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void) xcsf;
    l->n_inputs = prev->n_outputs;
    l->n_outputs = prev->n_outputs;
    l->max_outputs = prev->n_outputs;
    free_layer_arrays(l);
    malloc_layer_arrays(l);
}

double *
neural_layer_noise_output(const struct XCSF *xcsf, const LAYER *l)
{
    (void) xcsf;
    return l->output;
}

void
neural_layer_noise_print(const struct XCSF *xcsf, const LAYER *l,
                         _Bool print_weights)
{
    (void) xcsf;
    (void) print_weights;
    printf("noise in = %d, out = %d, prob = %f, stdev = %f\n", l->n_inputs,
           l->n_outputs, l->probability, l->scale);
}

size_t
neural_layer_noise_save(const struct XCSF *xcsf, const LAYER *l, FILE *fp)
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
neural_layer_noise_load(const struct XCSF *xcsf, LAYER *l, FILE *fp)
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
