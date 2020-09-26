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
 * @file neural_layer_recurrent.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a recurrent layer of perceptrons.
 * @details Fully-connected, stateful, and with a step of 1.
 */

#include "neural_layer_recurrent.h"
#include "blas.h"
#include "neural_activations.h"
#include "neural_layer_connected.h"
#include "sam.h"
#include "utils.h"

#define N_MU (6) //!< Number of mutation rates applied to a recurrent layer
static const int MU_TYPE[N_MU] = {
    SAM_RATE_SELECT, //!< Rate of gradient descent mutation
    SAM_RATE_SELECT, //!< Rate of neuron growth / removal
    SAM_RATE_SELECT, //!< Weight enabling mutation rate
    SAM_RATE_SELECT, //!< Weight disabling mutation rate
    SAM_RATE_SELECT, //!< Weight magnitude mutation
    SAM_RATE_SELECT //!< Activation function mutation rate
}; //<! Self-adaptation method

static void
malloc_layer_arrays(struct Layer *l)
{
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX) {
        printf("neural_layer_recurrent: malloc() invalid size\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->state = calloc(l->n_outputs, sizeof(double));
    l->prev_state = calloc(l->n_outputs, sizeof(double));
    l->mu = malloc(sizeof(double) * N_MU);
}

static void
free_layer_arrays(const struct Layer *l)
{
    free(l->state);
    free(l->prev_state);
    free(l->mu);
}

static void
set_layer_n_active(struct Layer *l)
{
    l->n_active = l->input_layer->n_active + l->self_layer->n_active +
        l->output_layer->n_active;
}

static void
set_layer_n_weights(struct Layer *l)
{
    l->n_weights = l->input_layer->n_weights + l->self_layer->n_weights +
        l->output_layer->n_weights;
}

static void
set_layer_n_biases(struct Layer *l)
{
    l->n_biases = l->input_layer->n_biases + l->self_layer->n_biases +
        l->output_layer->n_biases;
}

static _Bool
mutate_eta(const struct XCSF *xcsf, struct Layer *l)
{
    if ((l->options & LAYER_EVOLVE_ETA) &&
        layer_mutate_eta(xcsf, l, l->mu[0])) {
        l->input_layer->eta = l->eta;
        l->self_layer->eta = l->eta;
        l->output_layer->eta = l->eta;
        return true;
    }
    return false;
}

static _Bool
mutate_neurons(const struct XCSF *xcsf, struct Layer *l)
{
    if (l->options & LAYER_EVOLVE_NEURONS) {
        const int n = layer_mutate_neurons(xcsf, l->self_layer, l->mu[1]);
        if (n != 0) {
            layer_add_neurons(l->input_layer, n);
            layer_add_neurons(l->self_layer, n);
            layer_add_neurons(l->output_layer, n);
            layer_resize(xcsf, l->self_layer, l->input_layer);
            layer_resize(xcsf, l->output_layer, l->input_layer);
            l->n_outputs = l->output_layer->n_outputs;
            l->output = l->output_layer->output;
            l->delta = l->output_layer->delta;
            l->state = realloc(l->state, l->n_outputs * sizeof(double));
            l->prev_state =
                realloc(l->prev_state, l->n_outputs * sizeof(double));
            set_layer_n_weights(l);
            set_layer_n_biases(l);
            set_layer_n_active(l);
            return true;
        }
    }
    return false;
}

static _Bool
mutate_connectivity(struct Layer *l)
{
    _Bool mod = false;
    if (l->options & LAYER_EVOLVE_CONNECT) {
        if (layer_mutate_connectivity(l->input_layer, l->mu[2], l->mu[3])) {
            mod = true;
        }
        if (layer_mutate_connectivity(l->self_layer, l->mu[2], l->mu[3])) {
            mod = true;
        }
        if (layer_mutate_connectivity(l->output_layer, l->mu[2], l->mu[3])) {
            mod = true;
        }
        set_layer_n_active(l);
    }
    return mod;
}

static _Bool
mutate_weights(struct Layer *l)
{
    _Bool mod = false;
    if (l->options & LAYER_EVOLVE_WEIGHTS) {
        if (layer_mutate_weights(l->input_layer, l->mu[4])) {
            mod = true;
        }
        if (layer_mutate_weights(l->self_layer, l->mu[4])) {
            mod = true;
        }
        if (layer_mutate_weights(l->output_layer, l->mu[4])) {
            mod = true;
        }
    }
    return mod;
}

static _Bool
mutate_functions(struct Layer *l)
{
    if (l->options & LAYER_EVOLVE_FUNCTIONS &&
        layer_mutate_functions(l, l->mu[5])) {
        l->output_layer->function = l->function;
        return true;
    }
    return false;
}

/**
 * @brief Creates and initialises a recurrent layer.
 * @param xcsf The XCSF data structure.
 * @param n_inputs The number of inputs.
 * @param n_init The initial number of neurons.
 * @param n_max The maximum number of neurons.
 * @param f The activation function.
 * @param o The bitwise options specifying which operations can be performed.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_recurrent_init(const struct XCSF *xcsf, const int n_inputs,
                            const int n_init, const int n_max, const int f,
                            const uint32_t o)
{
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = RECURRENT;
    l->layer_vptr = &layer_recurrent_vtbl;
    l->options = o;
    l->function = f;
    l->n_inputs = n_inputs;
    l->n_outputs = n_init;
    l->max_outputs = n_max;
    l->input_layer =
        neural_layer_connected_init(xcsf, n_inputs, n_init, n_max, LINEAR, o);
    l->self_layer =
        neural_layer_connected_init(xcsf, n_init, n_init, n_max, LINEAR, o);
    l->output_layer =
        neural_layer_connected_init(xcsf, n_init, n_init, n_max, f, o);
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;
    l->eta = l->input_layer->eta;
    l->self_layer->eta = l->eta;
    l->output_layer->eta = l->eta;
    set_layer_n_biases(l);
    set_layer_n_weights(l);
    set_layer_n_active(l);
    malloc_layer_arrays(l);
    sam_init(l->mu, N_MU, MU_TYPE);
    return l;
}

struct Layer *
neural_layer_recurrent_copy(const struct XCSF *xcsf, const struct Layer *src)
{
    if (src->layer_type != RECURRENT) {
        printf("neural_layer_recurrent_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->options = src->options;
    l->function = src->function;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->n_active = src->n_active;
    l->eta = src->eta;
    l->max_outputs = src->max_outputs;
    l->input_layer = layer_copy(xcsf, src->input_layer);
    l->self_layer = layer_copy(xcsf, src->self_layer);
    l->output_layer = layer_copy(xcsf, src->output_layer);
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;
    malloc_layer_arrays(l);
    memcpy(l->mu, src->mu, sizeof(double) * N_MU);
    memcpy(l->prev_state, src->prev_state, sizeof(double) * src->n_outputs);
    return l;
}

void
neural_layer_recurrent_free(const struct XCSF *xcsf, const struct Layer *l)
{
    layer_free(xcsf, l->input_layer);
    layer_free(xcsf, l->self_layer);
    layer_free(xcsf, l->output_layer);
    free(l->input_layer);
    free(l->self_layer);
    free(l->output_layer);
    free_layer_arrays(l);
}

void
neural_layer_recurrent_rand(const struct XCSF *xcsf, struct Layer *l)
{
    layer_rand(xcsf, l->input_layer);
    layer_rand(xcsf, l->self_layer);
    layer_rand(xcsf, l->output_layer);
}

void
neural_layer_recurrent_forward(const struct XCSF *xcsf, const struct Layer *l,
                               const double *input)
{
    memcpy(l->prev_state, l->state, sizeof(double) * l->n_outputs);
    layer_forward(xcsf, l->input_layer, input);
    layer_forward(xcsf, l->self_layer, l->output_layer->output);
    memcpy(l->state, l->input_layer->output, sizeof(double) * l->n_outputs);
    blas_axpy(l->n_outputs, 1, l->self_layer->output, 1, l->state, 1);
    layer_forward(xcsf, l->output_layer, l->state);
}

void
neural_layer_recurrent_backward(const struct XCSF *xcsf, const struct Layer *l,
                                const double *input, double *delta)
{
    memset(l->input_layer->delta, 0, sizeof(double) * l->n_outputs);
    memset(l->self_layer->delta, 0, sizeof(double) * l->n_outputs);
    layer_backward(xcsf, l->output_layer, l->state, l->self_layer->delta);
    memcpy(l->input_layer->delta, l->self_layer->delta,
           sizeof(double) * l->n_outputs);
    layer_backward(xcsf, l->self_layer, l->prev_state, 0);
    layer_backward(xcsf, l->input_layer, input, delta);
}

void
neural_layer_recurrent_update(const struct XCSF *xcsf, const struct Layer *l)
{
    if (l->options & LAYER_SGD_WEIGHTS) {
        layer_update(xcsf, l->input_layer);
        layer_update(xcsf, l->self_layer);
        layer_update(xcsf, l->output_layer);
    }
}

void
neural_layer_recurrent_resize(const struct XCSF *xcsf, struct Layer *l,
                              const struct Layer *prev)
{
    layer_resize(xcsf, l->input_layer, prev);
    l->n_inputs = l->input_layer->n_inputs;
    l->n_active = l->input_layer->n_active + l->self_layer->n_active +
        l->output_layer->n_active;
}

double *
neural_layer_recurrent_output(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    return l->output;
}

_Bool
neural_layer_recurrent_mutate(const struct XCSF *xcsf, struct Layer *l)
{
    sam_adapt(l->mu, N_MU, MU_TYPE);
    _Bool mod = false;
    mod = mutate_eta(xcsf, l) ? true : mod;
    mod = mutate_neurons(xcsf, l) ? true : mod;
    mod = mutate_connectivity(l) ? true : mod;
    mod = mutate_weights(l) ? true : mod;
    mod = mutate_functions(l) ? true : mod;
    return mod;
}

void
neural_layer_recurrent_print(const struct XCSF *xcsf, const struct Layer *l,
                             const _Bool print_weights)
{
    printf("recurrent %s, in = %d, out = %d\n",
           neural_activation_string(l->function), l->n_inputs, l->n_outputs);
    if (print_weights) {
        printf("recurrent input layer:\n");
        layer_print(xcsf, l->input_layer, print_weights);
        printf("recurrent self layer:\n");
        layer_print(xcsf, l->self_layer, print_weights);
        printf("recurrent output layer:\n");
        layer_print(xcsf, l->output_layer, print_weights);
    }
}

size_t
neural_layer_recurrent_save(const struct XCSF *xcsf, const struct Layer *l,
                            FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    s += fwrite(l->state, sizeof(double), l->n_outputs, fp);
    s += fwrite(l->prev_state, sizeof(double), l->n_outputs, fp);
    s += layer_save(xcsf, l->input_layer, fp);
    s += layer_save(xcsf, l->self_layer, fp);
    s += layer_save(xcsf, l->output_layer, fp);
    return s;
}

size_t
neural_layer_recurrent_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    malloc_layer_arrays(l);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    s += fread(l->state, sizeof(double), l->n_outputs, fp);
    s += fread(l->prev_state, sizeof(double), l->n_outputs, fp);
    s += layer_load(xcsf, l->input_layer, fp);
    s += layer_load(xcsf, l->self_layer, fp);
    s += layer_load(xcsf, l->output_layer, fp);
    return s;
}
