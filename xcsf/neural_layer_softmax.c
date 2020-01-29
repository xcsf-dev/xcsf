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
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_softmax.h"
 
LAYER* neural_layer_softmax_init(const XCSF *xcsf, int in, double temp)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = SOFTMAX;
    l->layer_vptr = &layer_softmax_vtbl;
    l->temp = temp;
    l->num_inputs = in;
    l->num_outputs = in;
    l->max_outputs = in;
    l->options = 0;
    l->eta = 0;
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = calloc(l->num_inputs, sizeof(double));
    return l;
}

LAYER* neural_layer_softmax_copy(const XCSF *xcsf, const LAYER *from)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = from->layer_type;
    l->layer_vptr = from->layer_vptr;
    l->temp = from->temp;
    l->num_inputs = from->num_inputs;
    l->num_outputs = from->num_outputs;
    l->max_outputs = from->max_outputs;
    l->options = from->options;
    l->eta = 0;
    l->output = calloc(from->num_inputs, sizeof(double));
    l->delta = calloc(from->num_inputs, sizeof(double));
    return l;
}

void neural_layer_softmax_rand(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf; (void)l;
}

void neural_layer_softmax_forward(const XCSF *xcsf, const LAYER *l, const double *input)
{
    (void)xcsf;
    double largest = input[0];
    for(int i = 1; i < l->num_inputs; i++) {
        if(input[i] > largest) {
            largest = input[i];
        }
    }
    double sum = 0;
    for(int i = 0; i < l->num_inputs; i++) {
        double e = exp(input[i]/l->temp - largest/l->temp);
        sum += e;
        l->output[i] = e;
    }
    for(int i = 0; i < l->num_inputs; i++) {
        l->output[i] /= sum;
    }                                     
}

void neural_layer_softmax_backward(const XCSF *xcsf, const LAYER *l, const NET *net)
{
    (void)xcsf;
    for(int i = 0; i < l->num_inputs; i++) {
        net->delta[i] += l->delta[i];
    }
}

void neural_layer_softmax_update(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf; (void)l;
}

void neural_layer_softmax_print(const XCSF *xcsf, const LAYER *l, _Bool print_weights)
{
    (void)xcsf; (void)print_weights;
    printf("softmax in = %d, out = %d, temp = %f\n", 
            l->num_inputs, l->num_outputs, l->temp);
}

_Bool neural_layer_softmax_mutate(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
    return false;
}

void neural_layer_softmax_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    (void)xcsf;
    l->num_inputs = prev->num_outputs;
    l->num_outputs = prev->num_outputs;
    l->max_outputs = prev->num_outputs;
    free(l->output);
    free(l->delta);
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = calloc(l->num_inputs, sizeof(double));
}

void neural_layer_softmax_free(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    free(l->output);
    free(l->delta);
}

double* neural_layer_softmax_output(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    return l->output;
}
 
size_t neural_layer_softmax_save(const XCSF *xcsf, const LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&l->num_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->num_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->temp, sizeof(double), 1, fp);
    return s;
}

size_t neural_layer_softmax_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&l->num_inputs, sizeof(int), 1, fp);
    s += fread(&l->num_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->temp, sizeof(double), 1, fp);
    l->options = 0;
    l->eta = 0;
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = calloc(l->num_inputs, sizeof(double));
    return s;
}
