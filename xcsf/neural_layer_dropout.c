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
#include "neural_layer_dropout.h"

LAYER *neural_layer_dropout_init(const XCSF *xcsf, int in, double prob)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = DROPOUT;
    l->layer_vptr = &layer_dropout_vtbl;
    l->num_inputs = in;
    l->num_outputs = in;
    l->max_outputs = in;
    l->options = 0;
    l->probability = prob;
    l->scale = 1./(1.-prob);
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = malloc(l->num_inputs * sizeof(double));
    l->rand = malloc(l->num_inputs * sizeof(double));
    return l;
}

LAYER *neural_layer_dropout_copy(const XCSF *xcsf, LAYER *from)
{
    (void)xcsf;
    LAYER *l = malloc(sizeof(LAYER));
    l->layer_type = from->layer_type;
    l->layer_vptr = from->layer_vptr;
    l->num_inputs = from->num_inputs;
    l->num_outputs = from->num_inputs;
    l->max_outputs = from->max_outputs;
    l->options = from->options;
    l->probability = from->probability;
    l->scale = from->scale;
    l->output = calloc(from->num_inputs, sizeof(double));
    l->delta = malloc(from->num_inputs * sizeof(double));
    l->rand = malloc(from->num_inputs * sizeof(double));
    return l;
}
 
void neural_layer_dropout_free(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    free(l->output);
    free(l->delta);
    free(l->rand);
}

void neural_layer_dropout_rand(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}
 
void neural_layer_dropout_forward(const XCSF *xcsf, LAYER *l, const double *input)
{
    if(!xcsf->train) {
        for(int i = 0; i < l->num_inputs; i++) {
            l->output[i] = input[i];
        }
    }
    else {
        for(int i = 0; i < l->num_inputs; i++) {
            l->rand[i] = rand_uniform(0,1);
            if(l->rand[i] < l->probability) {
                l->output[i] = 0;
            }
            else {
                l->output[i] = input[i] * l->scale;
            }
        }
    }
}

void neural_layer_dropout_backward(const XCSF *xcsf, LAYER *l, NET *net)
{
    (void)xcsf;
    if(!net->delta) {
        return;
    }
    for(int i = 0; i < l->num_inputs; i++) {
        if(l->rand[i] < l->probability) {
            net->delta[i] = 0;
        }
        else {
            net->delta[i] += l->delta[i] * l->scale;
        }
    }
}

void neural_layer_dropout_update(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
}

_Bool neural_layer_dropout_mutate(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf; (void)l;
    return false;
}

void neural_layer_dropout_resize(const XCSF *xcsf, LAYER *l, LAYER *prev)
{
    (void)xcsf;
    l->num_inputs = prev->num_outputs;
    l->num_outputs = prev->num_outputs;
    l->max_outputs = prev->num_outputs;
    free(l->output);
    free(l->delta);
    free(l->rand);
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = calloc(l->num_inputs, sizeof(double));
    l->rand = calloc(l->num_inputs, sizeof(double));
}

double *neural_layer_dropout_output(const XCSF *xcsf, LAYER *l)
{
    (void)xcsf;
    return l->output;
}

void neural_layer_dropout_print(const XCSF *xcsf, LAYER *l, _Bool print_weights)
{
    (void)xcsf; (void)print_weights;
    printf("dropout in = %d, out = %d prob = %f\n",
            l->num_inputs, l->num_outputs, l->probability);
}
 
size_t neural_layer_dropout_save(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&l->num_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->num_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->probability, sizeof(double), 1, fp);
    s += fwrite(&l->scale, sizeof(double), 1, fp);
    return s;
}

size_t neural_layer_dropout_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&l->num_inputs, sizeof(int), 1, fp);
    s += fread(&l->num_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->probability, sizeof(double), 1, fp);
    s += fread(&l->scale, sizeof(double), 1, fp);
    l->options = 0;
    l->output = calloc(l->num_inputs, sizeof(double));
    l->delta = malloc(l->num_inputs * sizeof(double));
    l->rand = malloc(l->num_inputs * sizeof(double));
    return s;
}
