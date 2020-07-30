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
 * @file neural_layer_lstm.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a long short-term memory layer.
 * @details Stateful, and with a step of 1.
 * Typically the output activation is TANH and recurrent activation LOGISTIC.
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
#include "neural_layer_connected.h"
#include "neural_layer_lstm.h"

#define N_MU 5 //!< Number of mutation rates applied to a lstm layer

static _Bool mutate_connectivity(LAYER *l);
static _Bool mutate_eta(const XCSF *xcsf, LAYER *l);
static _Bool mutate_neurons(const XCSF *xcsf, LAYER *l);
static _Bool mutate_weights(LAYER *l);
static void calloc_layer_arrays(LAYER *l);
static void free_layer_arrays(const LAYER *l);
static void reset_layer_deltas(const LAYER *l);
static void set_eta(LAYER *l);
static void set_layer_n_active(LAYER *l);
static void set_layer_n_weights(LAYER *l);
static void set_layer_n_biases(LAYER *l);

LAYER *neural_layer_lstm_init(const XCSF *xcsf, int in, int n_init, int n_max, int f,
                              int rf, uint32_t o)
{
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = LSTM;
    l->layer_vptr = &layer_lstm_vtbl;
    l->options = o;
    l->function = f;
    l->recurrent_function = rf;
    l->n_inputs = in;
    l->n_outputs = n_init;
    l->max_outputs = n_max;
    l->uf = neural_layer_connected_init(xcsf, in, n_init, n_max, LINEAR, o);
    l->ui = neural_layer_connected_init(xcsf, in, n_init, n_max, LINEAR, o);
    l->ug = neural_layer_connected_init(xcsf, in, n_init, n_max, LINEAR, o);
    l->uo = neural_layer_connected_init(xcsf, in, n_init, n_max, LINEAR, o);
    l->wf = neural_layer_connected_init(xcsf, n_init, n_init, n_max, LINEAR, o);
    l->wi = neural_layer_connected_init(xcsf, n_init, n_init, n_max, LINEAR, o);
    l->wg = neural_layer_connected_init(xcsf, n_init, n_init, n_max, LINEAR, o);
    l->wo = neural_layer_connected_init(xcsf, n_init, n_init, n_max, LINEAR, o);
    set_layer_n_biases(l);
    set_layer_n_weights(l);
    set_layer_n_active(l);
    calloc_layer_arrays(l);
    l->mu = malloc(N_MU * sizeof(double));
    sam_init(xcsf, l->mu, N_MU);
    set_eta(l);
    return l;
}

static void set_layer_n_weights(LAYER *l)
{
    l->n_weights = l->uf->n_weights;
    l->n_weights += l->ui->n_weights;
    l->n_weights += l->ug->n_weights;
    l->n_weights += l->uo->n_weights;
    l->n_weights += l->wf->n_weights;
    l->n_weights += l->wi->n_weights;
    l->n_weights += l->wg->n_weights;
    l->n_weights += l->wo->n_weights;
}

static void set_layer_n_biases(LAYER *l)
{
    l->n_biases = l->uf->n_biases;
    l->n_biases += l->ui->n_biases;
    l->n_biases += l->ug->n_biases;
    l->n_biases += l->uo->n_biases;
    l->n_biases += l->wf->n_biases;
    l->n_biases += l->wi->n_biases;
    l->n_biases += l->wg->n_biases;
    l->n_biases += l->wo->n_biases;
}

static void set_layer_n_active(LAYER *l)
{
    l->n_active = l->uf->n_active;
    l->n_active += l->ui->n_active;
    l->n_active += l->ug->n_active;
    l->n_active += l->uo->n_active;
    l->n_active += l->wf->n_active;
    l->n_active += l->wi->n_active;
    l->n_active += l->wg->n_active;
    l->n_active += l->wo->n_active;
}

static void calloc_layer_arrays(LAYER *l)
{
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->state = calloc(l->n_outputs, sizeof(double));
    l->prev_state = calloc(l->n_outputs, sizeof(double));
    l->prev_cell = calloc(l->n_outputs, sizeof(double));
    l->cell = calloc(l->n_outputs, sizeof(double));
    l->f = calloc(l->n_outputs, sizeof(double));
    l->i = calloc(l->n_outputs, sizeof(double));
    l->g = calloc(l->n_outputs, sizeof(double));
    l->o = calloc(l->n_outputs, sizeof(double));
    l->c = calloc(l->n_outputs, sizeof(double));
    l->h = calloc(l->n_outputs, sizeof(double));
    l->temp = calloc(l->n_outputs, sizeof(double));
    l->temp2 = calloc(l->n_outputs, sizeof(double));
    l->temp3 = calloc(l->n_outputs, sizeof(double));
    l->dc = calloc(l->n_outputs, sizeof(double));
}

static void free_layer_arrays(const LAYER *l)
{
    free(l->delta);
    free(l->output);
    free(l->state);
    free(l->prev_state);
    free(l->prev_cell);
    free(l->cell);
    free(l->f);
    free(l->i);
    free(l->g);
    free(l->o);
    free(l->c);
    free(l->h);
    free(l->temp);
    free(l->temp2);
    free(l->temp3);
    free(l->dc);
}

static void set_eta(LAYER *l)
{
    l->eta = l->uf->eta;
    l->ui->eta = l->eta;
    l->ug->eta = l->eta;
    l->uo->eta = l->eta;
    l->wf->eta = l->eta;
    l->wi->eta = l->eta;
    l->wg->eta = l->eta;
    l->wo->eta = l->eta;
}

LAYER *neural_layer_lstm_copy(const XCSF *xcsf, const LAYER *src)
{
    LAYER *l = malloc(sizeof(LAYER));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->function = src->function;
    l->recurrent_function = src->recurrent_function;
    l->options = src->options;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->n_weights = src->n_weights;
    l->n_biases = src->n_biases;
    l->n_active = src->n_active;
    l->eta = src->eta;
    l->max_outputs = src->max_outputs;
    calloc_layer_arrays(l);
    l->uf = layer_copy(xcsf, src->uf);
    l->ui = layer_copy(xcsf, src->ui);
    l->ug = layer_copy(xcsf, src->ug);
    l->uo = layer_copy(xcsf, src->uo);
    l->wf = layer_copy(xcsf, src->wf);
    l->wi = layer_copy(xcsf, src->wi);
    l->wg = layer_copy(xcsf, src->wg);
    l->wo = layer_copy(xcsf, src->wo);
    l->mu = malloc(N_MU * sizeof(double));
    memcpy(l->mu, src->mu, N_MU * sizeof(double));
    return l;
}

void neural_layer_lstm_free(const XCSF *xcsf, const LAYER *l)
{
    layer_free(xcsf, l->uf);
    layer_free(xcsf, l->ui);
    layer_free(xcsf, l->ug);
    layer_free(xcsf, l->uo);
    layer_free(xcsf, l->wf);
    layer_free(xcsf, l->wi);
    layer_free(xcsf, l->wg);
    layer_free(xcsf, l->wo);
    free(l->uf);
    free(l->ui);
    free(l->ug);
    free(l->uo);
    free(l->wf);
    free(l->wi);
    free(l->wg);
    free(l->wo);
    free_layer_arrays(l);
    free(l->mu);
}

void neural_layer_lstm_rand(const XCSF *xcsf, LAYER *l)
{
    layer_rand(xcsf, l->uf);
    layer_rand(xcsf, l->ui);
    layer_rand(xcsf, l->ug);
    layer_rand(xcsf, l->uo);
    layer_rand(xcsf, l->wf);
    layer_rand(xcsf, l->wi);
    layer_rand(xcsf, l->wg);
    layer_rand(xcsf, l->wo);
}

void neural_layer_lstm_forward(const XCSF *xcsf, const LAYER *l, const double *input)
{
    layer_forward(xcsf, l->uf, input);
    layer_forward(xcsf, l->ui, input);
    layer_forward(xcsf, l->ug, input);
    layer_forward(xcsf, l->uo, input);
    input = l->h;
    layer_forward(xcsf, l->wf, input);
    layer_forward(xcsf, l->wi, input);
    layer_forward(xcsf, l->wg, input);
    layer_forward(xcsf, l->wo, input);
    memcpy(l->f, l->wf->output, l->n_outputs * sizeof(double));
    blas_axpy(l->n_outputs, 1, l->uf->output, 1, l->f, 1);
    memcpy(l->i, l->wi->output, l->n_outputs * sizeof(double));
    blas_axpy(l->n_outputs, 1, l->ui->output, 1, l->i, 1);
    memcpy(l->g, l->wg->output, l->n_outputs * sizeof(double));
    blas_axpy(l->n_outputs, 1, l->ug->output, 1, l->g, 1);
    memcpy(l->o, l->wo->output, l->n_outputs * sizeof(double));
    blas_axpy(l->n_outputs, 1, l->uo->output, 1, l->o, 1);
    neural_activate_array(l->f, l->f, l->n_outputs, l->recurrent_function);
    neural_activate_array(l->i, l->i, l->n_outputs, l->recurrent_function);
    neural_activate_array(l->g, l->g, l->n_outputs, l->function);
    neural_activate_array(l->o, l->o, l->n_outputs, l->recurrent_function);
    memcpy(l->temp, l->i, l->n_outputs * sizeof(double));
    blas_mul(l->n_outputs, l->g, 1, l->temp, 1);
    blas_mul(l->n_outputs, l->f, 1, l->c, 1);
    blas_axpy(l->n_outputs, 1, l->temp, 1, l->c, 1);
    memcpy(l->h, l->c, l->n_outputs * sizeof(double));
    neural_activate_array(l->h, l->h, l->n_outputs, l->function);
    blas_mul(l->n_outputs, l->o, 1, l->h, 1);
    memcpy(l->cell, l->c, l->n_outputs * sizeof(double));
    memcpy(l->output, l->h, l->n_outputs * sizeof(double));
}

static void reset_layer_deltas(const LAYER *l)
{
    size_t size = l->n_outputs * sizeof(double);
    memset(l->wf->delta, 0, size);
    memset(l->wi->delta, 0, size);
    memset(l->wg->delta, 0, size);
    memset(l->wo->delta, 0, size);
    memset(l->uf->delta, 0, size);
    memset(l->ui->delta, 0, size);
    memset(l->ug->delta, 0, size);
    memset(l->uo->delta, 0, size);
}

void neural_layer_lstm_backward(const XCSF *xcsf, const LAYER *l, const double *input,
                                double *delta)
{
    reset_layer_deltas(l);
    memcpy(l->temp3, l->delta, l->n_outputs * sizeof(double));
    memcpy(l->temp, l->c, l->n_outputs * sizeof(double));
    neural_activate_array(l->temp, l->temp, l->n_outputs, l->function);
    memcpy(l->temp2, l->temp3, l->n_outputs * sizeof(double));
    blas_mul(l->n_outputs, l->o, 1, l->temp2, 1);
    neural_gradient_array(l->temp, l->temp2, l->n_outputs, l->function);
    blas_axpy(l->n_outputs, 1, l->dc, 1, l->temp2, 1);
    memcpy(l->temp, l->c, l->n_outputs * sizeof(double));
    neural_activate_array(l->temp, l->temp, l->n_outputs, l->function);
    blas_mul(l->n_outputs, l->temp3, 1, l->temp, 1);
    neural_gradient_array(l->o, l->temp, l->n_outputs, l->recurrent_function);
    memcpy(l->wo->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->wo, l->prev_state, 0);
    memcpy(l->uo->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->uo, input, delta);
    memcpy(l->temp, l->temp2, l->n_outputs * sizeof(double));
    blas_mul(l->n_outputs, l->i, 1, l->temp, 1);
    neural_gradient_array(l->g, l->temp, l->n_outputs, l->function);
    memcpy(l->wg->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->wg, l->prev_state, 0);
    memcpy(l->ug->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->ug, input, delta);
    memcpy(l->temp, l->temp2, l->n_outputs * sizeof(double));
    blas_mul(l->n_outputs, l->g, 1, l->temp, 1);
    neural_gradient_array(l->i, l->temp, l->n_outputs, l->recurrent_function);
    memcpy(l->wi->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->wi, l->prev_state, 0);
    memcpy(l->ui->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->ui, input, delta);
    memcpy(l->temp, l->temp2, l->n_outputs * sizeof(double));
    blas_mul(l->n_outputs, l->prev_cell, 1, l->temp, 1);
    neural_gradient_array(l->f, l->temp, l->n_outputs, l->recurrent_function);
    memcpy(l->wf->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->wf, l->prev_state, 0);
    memcpy(l->uf->delta, l->temp, l->n_outputs * sizeof(double));
    layer_backward(xcsf, l->uf, input, delta);
    memcpy(l->temp, l->temp2, l->n_outputs * sizeof(double));
    blas_mul(l->n_outputs, l->f, 1, l->temp, 1);
    memcpy(l->dc, l->temp, l->n_outputs * sizeof(double));
}

void neural_layer_lstm_update(const XCSF *xcsf, const LAYER *l)
{
    if(l->options & LAYER_SGD_WEIGHTS) {
        layer_update(xcsf, l->wf);
        layer_update(xcsf, l->wi);
        layer_update(xcsf, l->wg);
        layer_update(xcsf, l->wo);
        layer_update(xcsf, l->uf);
        layer_update(xcsf, l->ui);
        layer_update(xcsf, l->ug);
        layer_update(xcsf, l->uo);
    }
}

void neural_layer_lstm_resize(const XCSF *xcsf, LAYER *l, const LAYER *prev)
{
    layer_resize(xcsf, l->uf, prev);
    layer_resize(xcsf, l->ui, prev);
    layer_resize(xcsf, l->ug, prev);
    layer_resize(xcsf, l->uo, prev);
    layer_resize(xcsf, l->uf, prev);
    l->n_inputs = prev->n_outputs;
    set_layer_n_weights(l);
    set_layer_n_biases(l);
    set_layer_n_active(l);
}

double *neural_layer_lstm_output(const XCSF *xcsf, const LAYER *l)
{
    (void)xcsf;
    return l->output;
}

_Bool neural_layer_lstm_mutate(const XCSF *xcsf, LAYER *l)
{
    sam_adapt(xcsf, l->mu, N_MU);
    _Bool mod = false;
    if((l->options & LAYER_EVOLVE_ETA) && mutate_eta(xcsf, l)) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_NEURONS) && mutate_neurons(xcsf, l)) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_CONNECT) && mutate_connectivity(l)) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_WEIGHTS) && mutate_weights(l)) {
        mod = true;
    }
    if((l->options & LAYER_EVOLVE_FUNCTIONS) && layer_mutate_functions(l, l->mu[4])) {
        mod = true;
    }
    return mod;
}

static _Bool mutate_eta(const XCSF *xcsf, LAYER *l)
{
    if(layer_mutate_eta(xcsf, l->uf, l->mu[0])) {
        set_eta(l);
        return true;
    }
    return false;
}

static _Bool mutate_neurons(const XCSF *xcsf, LAYER *l)
{
    int n = layer_mutate_neurons(xcsf, l->uf, l->mu[1]);
    if(n != 0) {
        layer_add_neurons(l->uf, n);
        layer_add_neurons(l->ui, n);
        layer_add_neurons(l->ug, n);
        layer_add_neurons(l->uo, n);
        layer_add_neurons(l->wf, n);
        layer_add_neurons(l->wi, n);
        layer_add_neurons(l->wg, n);
        layer_add_neurons(l->wo, n);
        layer_resize(xcsf, l->wf, l->uf);
        layer_resize(xcsf, l->wi, l->uf);
        layer_resize(xcsf, l->wg, l->uf);
        layer_resize(xcsf, l->wo, l->uf);
        l->n_outputs = l->uf->n_outputs;
        free_layer_arrays(l);
        calloc_layer_arrays(l);
        set_layer_n_weights(l);
        set_layer_n_biases(l);
        set_layer_n_active(l);
        return true;
    }
    return false;
}

static _Bool mutate_connectivity(LAYER *l)
{
    _Bool mod = false;
    mod = layer_mutate_connectivity(l->uf, l->mu[2]) ? true : mod;
    mod = layer_mutate_connectivity(l->ui, l->mu[2]) ? true : mod;
    mod = layer_mutate_connectivity(l->ug, l->mu[2]) ? true : mod;
    mod = layer_mutate_connectivity(l->uo, l->mu[2]) ? true : mod;
    mod = layer_mutate_connectivity(l->wf, l->mu[2]) ? true : mod;
    mod = layer_mutate_connectivity(l->wi, l->mu[2]) ? true : mod;
    mod = layer_mutate_connectivity(l->wg, l->mu[2]) ? true : mod;
    mod = layer_mutate_connectivity(l->wo, l->mu[2]) ? true : mod;
    set_layer_n_active(l);
    return mod;
}

static _Bool mutate_weights(LAYER *l)
{
    _Bool mod = false;
    mod = layer_mutate_weights(l->uf, l->mu[3]) ? true : mod;
    mod = layer_mutate_weights(l->ui, l->mu[3]) ? true : mod;
    mod = layer_mutate_weights(l->ug, l->mu[3]) ? true : mod;
    mod = layer_mutate_weights(l->uo, l->mu[3]) ? true : mod;
    mod = layer_mutate_weights(l->wf, l->mu[3]) ? true : mod;
    mod = layer_mutate_weights(l->wi, l->mu[3]) ? true : mod;
    mod = layer_mutate_weights(l->wg, l->mu[3]) ? true : mod;
    mod = layer_mutate_weights(l->wo, l->mu[3]) ? true : mod;
    return mod;
}

void neural_layer_lstm_print(const XCSF *xcsf, const LAYER *l, _Bool print_weights)
{
    printf("lstm, f = %s, rf = %s,  in = %d, out = %d\n",
           neural_activation_string(l->function),
           neural_activation_string(l->recurrent_function),
           l->n_inputs, l->n_outputs);
    if(print_weights) {
        printf("uf layer:\n");
        layer_print(xcsf, l->uf, print_weights);
        printf("ui layer:\n");
        layer_print(xcsf, l->ui, print_weights);
        printf("ug layer:\n");
        layer_print(xcsf, l->ug, print_weights);
        printf("uo layer:\n");
        layer_print(xcsf, l->uo, print_weights);
        printf("wf layer:\n");
        layer_print(xcsf, l->wf, print_weights);
        printf("wi layer:\n");
        layer_print(xcsf, l->wi, print_weights);
        printf("wg layer:\n");
        layer_print(xcsf, l->wg, print_weights);
        printf("wo layer:\n");
        layer_print(xcsf, l->wo, print_weights);
    }
}

size_t neural_layer_lstm_save(const XCSF *xcsf, const LAYER *l, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_weights, sizeof(int), 1, fp);
    s += fwrite(&l->n_biases, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    s += fwrite(l->state, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->prev_state, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->cell, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->f, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->i, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->g, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->o, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->c, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->h, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->temp, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->temp2, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->temp3, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->dc, sizeof(double), l->n_outputs, fp);
    s += layer_save(xcsf, l->uf, fp);
    s += layer_save(xcsf, l->ui, fp);
    s += layer_save(xcsf, l->ug, fp);
    s += layer_save(xcsf, l->uo, fp);
    s += layer_save(xcsf, l->wf, fp);
    s += layer_save(xcsf, l->wi, fp);
    s += layer_save(xcsf, l->wg, fp);
    s += layer_save(xcsf, l->wo, fp);
    return s;
}

size_t neural_layer_lstm_load(const XCSF *xcsf, LAYER *l, FILE *fp)
{
    size_t s = 0;
    layer_init(l);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_weights, sizeof(int), 1, fp);
    s += fread(&l->n_biases, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    if(l->n_inputs < 1 || l->n_outputs < 1 || l->n_biases < 1) {
        printf("neural_layer_lstm_load(): read error\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->mu = malloc(N_MU * sizeof(double));
    s += fread(l->mu, sizeof(double), N_MU, fp);
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->state = malloc(l->n_outputs * sizeof(double));
    l->prev_state = malloc(l->n_outputs * sizeof(double));
    l->prev_cell = malloc(l->n_outputs * sizeof(double));
    l->cell = malloc(l->n_outputs * sizeof(double));
    l->f = malloc(l->n_outputs * sizeof(double));
    l->i = malloc(l->n_outputs * sizeof(double));
    l->g = malloc(l->n_outputs * sizeof(double));
    l->o = malloc(l->n_outputs * sizeof(double));
    l->c = malloc(l->n_outputs * sizeof(double));
    l->h = malloc(l->n_outputs * sizeof(double));
    l->temp = malloc(l->n_outputs * sizeof(double));
    l->temp2 = malloc(l->n_outputs * sizeof(double));
    l->temp3 = malloc(l->n_outputs * sizeof(double));
    l->dc = malloc(l->n_outputs * sizeof(double));
    s += fread(l->f, sizeof(double), l->n_outputs, fp);
    s += fread(l->i, sizeof(double), l->n_outputs, fp);
    s += fread(l->g, sizeof(double), l->n_outputs, fp);
    s += fread(l->o, sizeof(double), l->n_outputs, fp);
    s += fread(l->c, sizeof(double), l->n_outputs, fp);
    s += fread(l->h, sizeof(double), l->n_outputs, fp);
    s += fread(l->temp, sizeof(double), l->n_outputs, fp);
    s += fread(l->temp2, sizeof(double), l->n_outputs, fp);
    s += fread(l->temp3, sizeof(double), l->n_outputs, fp);
    s += fread(l->dc, sizeof(double), l->n_outputs, fp);
    s += layer_load(xcsf, l->uf, fp);
    s += layer_load(xcsf, l->ui, fp);
    s += layer_load(xcsf, l->ug, fp);
    s += layer_load(xcsf, l->uo, fp);
    s += layer_load(xcsf, l->wf, fp);
    s += layer_load(xcsf, l->wi, fp);
    s += layer_load(xcsf, l->wg, fp);
    s += layer_load(xcsf, l->wo, fp);
    return s;
}
