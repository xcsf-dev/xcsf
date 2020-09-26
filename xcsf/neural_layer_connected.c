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
 * @file neural_layer_connected.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a fully-connected layer of perceptrons.
 */

#include "neural_layer_connected.h"
#include "blas.h"
#include "neural_activations.h"
#include "sam.h"
#include "utils.h"

#define N_MU (6) //!< Number of mutation rates applied to a connected layer
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
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX || l->n_weights < 1 ||
        l->n_weights > N_WEIGHTS_MAX) {
        printf("neural_layer_connected: malloc() invalid size\n");
        l->n_weights = 1;
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->biases = malloc(sizeof(double) * l->n_outputs);
    l->bias_updates = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->weight_updates = calloc(l->n_weights, sizeof(double));
    l->weight_active = malloc(sizeof(_Bool) * l->n_weights);
    l->weights = malloc(sizeof(double) * l->n_weights);
    l->mu = malloc(sizeof(double) * N_MU);
}

/**
 * @brief Creates and initialises a fully-connected layer.
 * @param xcsf The XCSF data structure.
 * @param n_inputs The number of inputs.
 * @param n_init The initial number of neurons.
 * @param n_max The maximum number of neurons.
 * @param f The activation function.
 * @param o The bitwise options specifying which operations can be performed.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_connected_init(const struct XCSF *xcsf, const int n_inputs,
                            const int n_init, const int n_max, const int f,
                            const uint32_t o)
{
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = CONNECTED;
    l->layer_vptr = &layer_connected_vtbl;
    l->options = o;
    l->function = f;
    l->n_inputs = n_inputs;
    l->n_outputs = n_init;
    l->max_outputs = n_max;
    l->n_weights = n_inputs * n_init;
    l->n_biases = l->n_outputs;
    l->n_active = l->n_weights;
    layer_init_eta(xcsf, l);
    malloc_layer_arrays(l);
    for (int i = 0; i < l->n_weights; ++i) {
        l->weights[i] = rand_normal(0, 0.1);
        l->weight_active[i] = true;
    }
    memset(l->biases, 0, sizeof(double) * l->n_biases);
    sam_init(l->mu, N_MU, MU_TYPE);
    return l;
}

void
neural_layer_connected_free(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    free(l->state);
    free(l->output);
    free(l->biases);
    free(l->bias_updates);
    free(l->delta);
    free(l->weight_updates);
    free(l->weight_active);
    free(l->weights);
    free(l->mu);
}

struct Layer *
neural_layer_connected_copy(const struct XCSF *xcsf, const struct Layer *src)
{
    (void) xcsf;
    if (src->layer_type != CONNECTED) {
        printf("neural_layer_connected_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->function = src->function;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->n_weights = src->n_weights;
    l->n_biases = src->n_biases;
    l->options = src->options;
    l->eta = src->eta;
    l->n_active = src->n_active;
    malloc_layer_arrays(l);
    memcpy(l->biases, src->biases, sizeof(double) * src->n_biases);
    memcpy(l->weights, src->weights, sizeof(double) * src->n_weights);
    memcpy(l->weight_active, src->weight_active,
           sizeof(_Bool) * src->n_weights);
    memcpy(l->mu, src->mu, sizeof(double) * N_MU);
    return l;
}

void
neural_layer_connected_rand(const struct XCSF *xcsf, struct Layer *l)
{
    layer_weight_rand(xcsf, l);
}

void
neural_layer_connected_forward(const struct XCSF *xcsf, const struct Layer *l,
                               const double *input)
{
    (void) xcsf;
    const int k = l->n_inputs;
    const int n = l->n_outputs;
    const double *a = input;
    const double *b = l->weights;
    double *c = l->state;
    memcpy(l->state, l->biases, sizeof(double) * l->n_outputs);
    blas_gemm(0, 1, 1, n, k, 1, a, k, b, k, 1, c, n);
    neural_activate_array(l->state, l->output, l->n_outputs, l->function);
}

void
neural_layer_connected_backward(const struct XCSF *xcsf, const struct Layer *l,
                                const double *input, double *delta)
{
    (void) xcsf;
    neural_gradient_array(l->state, l->delta, l->n_outputs, l->function);
    if (l->options & LAYER_SGD_WEIGHTS) {
        const int m = l->n_outputs;
        const int n = l->n_inputs;
        const double *a = l->delta;
        const double *b = input;
        double *c = l->weight_updates;
        blas_axpy(l->n_outputs, 1, l->delta, 1, l->bias_updates, 1);
        blas_gemm(1, 0, m, n, 1, 1, a, m, b, n, 1, c, n);
    }
    if (delta) {
        const int k = l->n_outputs;
        const int n = l->n_inputs;
        const double *a = l->delta;
        const double *b = l->weights;
        double *c = delta;
        blas_gemm(0, 0, 1, n, k, 1, a, k, b, n, 1, c, n);
    }
}

void
neural_layer_connected_update(const struct XCSF *xcsf, const struct Layer *l)
{
    if (l->options & LAYER_SGD_WEIGHTS) {
        blas_axpy(l->n_biases, l->eta, l->bias_updates, 1, l->biases, 1);
        blas_scal(l->n_biases, xcsf->PRED_MOMENTUM, l->bias_updates, 1);
        if (xcsf->PRED_DECAY > 0) {
            blas_axpy(l->n_weights, -(xcsf->PRED_DECAY), l->weights, 1,
                      l->weight_updates, 1);
        }
        blas_axpy(l->n_weights, l->eta, l->weight_updates, 1, l->weights, 1);
        blas_scal(l->n_weights, xcsf->PRED_MOMENTUM, l->weight_updates, 1);
        layer_weight_clamp(l);
    }
}

void
neural_layer_connected_resize(const struct XCSF *xcsf, struct Layer *l,
                              const struct Layer *prev)
{
    (void) xcsf;
    const int n_weights = prev->n_outputs * l->n_outputs;
    double *weights = malloc(sizeof(double) * n_weights);
    double *weight_updates = malloc(sizeof(double) * n_weights);
    _Bool *weight_active = malloc(sizeof(_Bool) * n_weights);
    for (int i = 0; i < l->n_outputs; ++i) {
        const int orig_offset = i * l->n_inputs;
        const int offset = i * prev->n_outputs;
        for (int j = 0; j < prev->n_outputs; ++j) {
            if (j < l->n_inputs) {
                weights[offset + j] = l->weights[orig_offset + j];
                weight_updates[offset + j] = l->weight_updates[orig_offset + j];
                weight_active[offset + j] = l->weight_active[orig_offset + j];
            } else {
                weights[offset + j] = rand_normal(0, 0.1);
                weight_updates[offset + j] = 0;
                weight_active[offset + j] = true;
            }
        }
    }
    free(l->weights);
    free(l->weight_updates);
    free(l->weight_active);
    l->weights = weights;
    l->weight_updates = weight_updates;
    l->weight_active = weight_active;
    l->n_weights = n_weights;
    l->n_inputs = prev->n_outputs;
    layer_calc_n_active(l);
    if (l->options & LAYER_EVOLVE_CONNECT) {
        layer_ensure_input_represention(l);
    }
}

_Bool
neural_layer_connected_mutate(const struct XCSF *xcsf, struct Layer *l)
{
    sam_adapt(l->mu, N_MU, MU_TYPE);
    _Bool mod = false;
    if ((l->options & LAYER_EVOLVE_ETA) &&
        layer_mutate_eta(xcsf, l, l->mu[0])) {
        mod = true;
    }
    if (l->options & LAYER_EVOLVE_NEURONS) {
        const int n = layer_mutate_neurons(xcsf, l, l->mu[1]);
        if (n != 0) {
            layer_add_neurons(l, n);
            mod = true;
        }
    }
    if ((l->options & LAYER_EVOLVE_CONNECT) &&
        layer_mutate_connectivity(l, l->mu[2], l->mu[3])) {
        layer_ensure_input_represention(l);
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_WEIGHTS) &&
        layer_mutate_weights(l, l->mu[4])) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_FUNCTIONS) &&
        layer_mutate_functions(l, l->mu[5])) {
        mod = true;
    }
    return mod;
}

double *
neural_layer_connected_output(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    return l->output;
}

void
neural_layer_connected_print(const struct XCSF *xcsf, const struct Layer *l,
                             const _Bool print_weights)
{
    (void) xcsf;
    printf("connected %s, in = %d, out = %d, ",
           neural_activation_string(l->function), l->n_inputs, l->n_outputs);
    layer_weight_print(l, print_weights);
    printf("\n");
}

size_t
neural_layer_connected_save(const struct XCSF *xcsf, const struct Layer *l,
                            FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_biases, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_weights, sizeof(int), 1, fp);
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(l->weights, sizeof(double), l->n_weights, fp);
    s += fwrite(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->n_biases, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->n_biases, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t
neural_layer_connected_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_biases, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_weights, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    malloc_layer_arrays(l);
    s += fread(l->weights, sizeof(double), l->n_weights, fp);
    s += fread(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fread(l->biases, sizeof(double), l->n_biases, fp);
    s += fread(l->bias_updates, sizeof(double), l->n_biases, fp);
    s += fread(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    return s;
}
