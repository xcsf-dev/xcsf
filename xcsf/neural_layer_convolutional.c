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
 * @file neural_layer_convolutional.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D convolutional layer.
 */

#include "neural_layer_convolutional.h"
#include "blas.h"
#include "image.h"
#include "neural_activations.h"
#include "sam.h"
#include "utils.h"

#define N_MU (5) //!< Number of mutation rates applied to a convolutional layer
static const int MU_TYPE[N_MU] = {
    SAM_RATE_SELECT, SAM_RATE_SELECT, SAM_RATE_SELECT, SAM_RATE_SELECT,
    SAM_RATE_SELECT
}; //<! Self-adaptation method

static size_t
get_workspace_size(const struct LAYER *l)
{
    int size = l->out_h * l->out_w * l->size * l->size * l->channels;
    if (size < 1) {
        printf("neural_layer_convolutional: workspace_size overflow\n");
        exit(EXIT_FAILURE);
    }
    return sizeof(double) * size;
}

static void
malloc_layer_arrays(struct LAYER *l)
{
    if (l->n_biases < 1 || l->n_biases > N_OUTPUTS_MAX || l->n_outputs < 1 ||
        l->n_outputs > N_OUTPUTS_MAX || l->n_weights < 1 ||
        l->n_weights > N_WEIGHTS_MAX || l->workspace_size < 1) {
        printf("neural_layer_convolutional: malloc() invalid size\n");
        l->n_biases = 1;
        l->n_outputs = 1;
        l->n_weights = 1;
        l->workspace_size = 1;
        exit(EXIT_FAILURE);
    }
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->weights = malloc(sizeof(double) * l->n_weights);
    l->biases = malloc(sizeof(double) * l->n_biases);
    l->bias_updates = calloc(l->n_biases, sizeof(double));
    l->weight_updates = calloc(l->n_weights, sizeof(double));
    l->weight_active = malloc(sizeof(_Bool) * l->n_weights);
    l->temp = malloc(l->workspace_size);
    l->mu = malloc(sizeof(double) * N_MU);
}

static int
convolutional_out_height(const struct LAYER *l)
{
    return (l->height + 2 * l->pad - l->size) / l->stride + 1;
}

static int
convolutional_out_width(const struct LAYER *l)
{
    return (l->width + 2 * l->pad - l->size) / l->stride + 1;
}

/**
 * @brief Creates and initialises a 2D convolutional layer.
 * @param xcsf The XCSF data structure.
 * @param h The input height.
 * @param w The input width.
 * @param c The number of input channels.
 * @param n_filters The number of kernel filters to apply.
 * @param kernel_size The length of the convolution window.
 * @param stride The stride length of the convolution.
 * @param pad The padding of the convolution.
 * @param f The activation function.
 * @param o The bitwise options specifying which operations can be performed.
 * @return A pointer to the new layer.
 */
struct LAYER *
neural_layer_convolutional_init(const struct XCSF *xcsf, int h, int w, int c,
                                int n_filters, int kernel_size, int stride,
                                int pad, int f, uint32_t o)
{
    struct LAYER *l = malloc(sizeof(struct LAYER));
    layer_init(l);
    l->layer_type = CONVOLUTIONAL;
    l->layer_vptr = &layer_convolutional_vtbl;
    l->options = o;
    l->function = f;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->n_filters = n_filters;
    l->stride = stride;
    l->size = kernel_size;
    l->pad = pad;
    l->n_biases = n_filters;
    l->n_weights = l->channels * n_filters * kernel_size * kernel_size;
    l->n_active = l->n_weights;
    l->out_h = convolutional_out_height(l);
    l->out_w = convolutional_out_width(l);
    l->out_c = n_filters;
    l->n_inputs = l->width * l->height * l->channels;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->workspace_size = get_workspace_size(l);
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
neural_layer_convolutional_free(const struct XCSF *xcsf, const struct LAYER *l)
{
    (void) xcsf;
    free(l->delta);
    free(l->state);
    free(l->output);
    free(l->weights);
    free(l->biases);
    free(l->bias_updates);
    free(l->weight_updates);
    free(l->weight_active);
    free(l->temp);
    free(l->mu);
}

struct LAYER *
neural_layer_convolutional_copy(const struct XCSF *xcsf,
                                const struct LAYER *src)
{
    (void) xcsf;
    if (src->layer_type != CONVOLUTIONAL) {
        printf("neural_layer_convolut_copy() incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct LAYER *l = malloc(sizeof(struct LAYER));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->options = src->options;
    l->function = src->function;
    l->height = src->height;
    l->width = src->width;
    l->channels = src->channels;
    l->n_filters = src->n_filters;
    l->stride = src->stride;
    l->size = src->size;
    l->pad = src->pad;
    l->n_weights = src->n_weights;
    l->n_active = src->n_active;
    l->out_h = src->out_h;
    l->out_w = src->out_w;
    l->out_c = src->out_c;
    l->n_outputs = src->n_outputs;
    l->n_inputs = src->n_inputs;
    l->max_outputs = src->max_outputs;
    l->n_biases = src->n_biases;
    l->eta = src->eta;
    l->workspace_size = src->workspace_size;
    malloc_layer_arrays(l);
    memcpy(l->weights, src->weights, sizeof(double) * src->n_weights);
    memcpy(l->weight_active, src->weight_active,
           sizeof(_Bool) * src->n_weights);
    memcpy(l->biases, src->biases, sizeof(double) * src->n_biases);
    memcpy(l->mu, src->mu, sizeof(double) * N_MU);
    return l;
}

void
neural_layer_convolutional_rand(const struct XCSF *xcsf, struct LAYER *l)
{
    layer_weight_rand(xcsf, l);
}

void
neural_layer_convolutional_forward(const struct XCSF *xcsf,
                                   const struct LAYER *l, const double *input)
{
    (void) xcsf;
    int m = l->n_filters;
    int k = l->size * l->size * l->channels;
    int n = l->out_w * l->out_h;
    const double *a = l->weights;
    double *b = l->temp;
    double *c = l->state;
    memset(l->state, 0, sizeof(double) * l->n_outputs);
    if (l->size == 1) {
        blas_gemm(0, 0, m, n, k, 1, a, k, input, n, 1, c, n);
    } else {
        im2col(input, l->channels, l->height, l->width, l->size, l->stride,
               l->pad, b);
        blas_gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }
    for (int i = 0; i < l->n_biases; ++i) {
        for (int j = 0; j < n; ++j) {
            l->state[i * n + j] += l->biases[i];
        }
    }
    neural_activate_array(l->state, l->output, l->n_outputs, l->function);
}

void
neural_layer_convolutional_backward(const struct XCSF *xcsf,
                                    const struct LAYER *l, const double *input,
                                    double *delta)
{
    (void) xcsf;
    int m = l->n_filters;
    int n = l->size * l->size * l->channels;
    int k = l->out_w * l->out_h;
    if (l->options & LAYER_SGD_WEIGHTS) {
        neural_gradient_array(l->state, l->delta, l->n_outputs, l->function);
        for (int i = 0; i < l->n_biases; ++i) {
            l->bias_updates[i] += blas_sum(l->delta + k * i, k);
        }
        const double *a = l->delta;
        double *b = l->temp;
        double *c = l->weight_updates;
        if (l->size == 1) {
            blas_gemm(0, 1, m, n, k, 1, a, k, input, k, 1, c, n);
        } else {
            im2col(input, l->channels, l->height, l->width, l->size, l->stride,
                   l->pad, b);
            blas_gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
        }
    }
    if (delta) {
        const double *a = l->weights;
        const double *b = l->delta;
        double *c = l->temp;
        if (l->size == 1) {
            c = delta;
        }
        blas_gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
        if (l->size != 1) {
            col2im(l->temp, l->channels, l->height, l->width, l->size,
                   l->stride, l->pad, delta);
        }
    }
}

void
neural_layer_convolutional_update(const struct XCSF *xcsf,
                                  const struct LAYER *l)
{
    if (l->options & LAYER_SGD_WEIGHTS) {
        blas_axpy(l->n_biases, l->eta, l->bias_updates, 1, l->biases, 1);
        blas_axpy(l->n_weights, l->eta, l->weight_updates, 1, l->weights, 1);
        blas_scal(l->n_biases, xcsf->PRED_MOMENTUM, l->bias_updates, 1);
        blas_scal(l->n_weights, xcsf->PRED_MOMENTUM, l->weight_updates, 1);
        layer_weight_clamp(l);
    }
}

void
neural_layer_convolutional_resize(const struct XCSF *xcsf, struct LAYER *l,
                                  const struct LAYER *prev)
{
    (void) xcsf;
    l->width = prev->out_w;
    l->height = prev->out_h;
    l->channels = prev->out_c;
    l->out_w = convolutional_out_width(l);
    l->out_h = convolutional_out_height(l);
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = l->width * l->height * l->channels;
    l->state = realloc(l->state, sizeof(double) * l->n_outputs);
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
    l->workspace_size = get_workspace_size(l);
}

_Bool
neural_layer_convolutional_mutate(const struct XCSF *xcsf, struct LAYER *l)
{
    sam_adapt(l->mu, N_MU, MU_TYPE);
    _Bool mod = false;
    if ((l->options & LAYER_EVOLVE_ETA) &&
        layer_mutate_eta(xcsf, l, l->mu[0])) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_CONNECT) &&
        layer_mutate_connectivity(l, l->mu[1], l->mu[2])) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_WEIGHTS) &&
        layer_mutate_weights(l, l->mu[3])) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_FUNCTIONS) &&
        layer_mutate_functions(l, l->mu[4])) {
        mod = true;
    }
    return mod;
}

double *
neural_layer_convolutional_output(const struct XCSF *xcsf,
                                  const struct LAYER *l)
{
    (void) xcsf;
    return l->output;
}

void
neural_layer_convolutional_print(const struct XCSF *xcsf, const struct LAYER *l,
                                 _Bool print_weights)
{
    (void) xcsf;
    printf("convolutional %s, in=%d, out=%d, filters=%d, size=%d, stride=%d, "
           "pad=%d",
           neural_activation_string(l->function), l->n_inputs, l->n_outputs,
           l->size, l->n_filters, l->stride, l->pad);
    layer_weight_print(l, print_weights);
    printf("\n");
}

size_t
neural_layer_convolutional_save(const struct XCSF *xcsf, const struct LAYER *l,
                                FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->height, sizeof(int), 1, fp);
    s += fwrite(&l->width, sizeof(int), 1, fp);
    s += fwrite(&l->channels, sizeof(int), 1, fp);
    s += fwrite(&l->n_filters, sizeof(int), 1, fp);
    s += fwrite(&l->stride, sizeof(int), 1, fp);
    s += fwrite(&l->size, sizeof(int), 1, fp);
    s += fwrite(&l->pad, sizeof(int), 1, fp);
    s += fwrite(&l->out_h, sizeof(int), 1, fp);
    s += fwrite(&l->out_w, sizeof(int), 1, fp);
    s += fwrite(&l->out_c, sizeof(int), 1, fp);
    s += fwrite(&l->n_biases, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_weights, sizeof(int), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(&l->workspace_size, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(l->weights, sizeof(double), l->n_weights, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fwrite(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->n_biases, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->n_filters, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    return s;
}

size_t
neural_layer_convolutional_load(const struct XCSF *xcsf, struct LAYER *l,
                                FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->height, sizeof(int), 1, fp);
    s += fread(&l->width, sizeof(int), 1, fp);
    s += fread(&l->channels, sizeof(int), 1, fp);
    s += fread(&l->n_filters, sizeof(int), 1, fp);
    s += fread(&l->stride, sizeof(int), 1, fp);
    s += fread(&l->size, sizeof(int), 1, fp);
    s += fread(&l->pad, sizeof(int), 1, fp);
    s += fread(&l->out_h, sizeof(int), 1, fp);
    s += fread(&l->out_w, sizeof(int), 1, fp);
    s += fread(&l->out_c, sizeof(int), 1, fp);
    s += fread(&l->n_biases, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_weights, sizeof(int), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    s += fread(&l->workspace_size, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    malloc_layer_arrays(l);
    s += fread(l->weights, sizeof(double), l->n_weights, fp);
    s += fread(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fread(l->weight_active, sizeof(_Bool), l->n_weights, fp);
    s += fread(l->biases, sizeof(double), l->n_biases, fp);
    s += fread(l->bias_updates, sizeof(double), l->n_biases, fp);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    return s;
}
