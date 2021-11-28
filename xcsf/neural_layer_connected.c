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
 * @date 2016--2021.
 * @brief An implementation of a fully-connected layer of perceptrons.
 */

#include "neural_layer_connected.h"
#include "blas.h"
#include "neural_activations.h"
#include "sam.h"
#include "utils.h"

#define N_MU (6) //!< Number of mutation rates applied to a connected layer

/**
 * @brief Self-adaptation method for mutating a connected layer.
 */
static const int MU_TYPE[N_MU] = {
    SAM_RATE_SELECT, //!< Rate of gradient descent mutation
    SAM_RATE_SELECT, //!< Rate of neuron growth / removal
    SAM_RATE_SELECT, //!< Weight enabling mutation rate
    SAM_RATE_SELECT, //!< Weight disabling mutation rate
    SAM_RATE_SELECT, //!< Weight magnitude mutation
    SAM_RATE_SELECT //!< Activation function mutation rate
};

/**
 * @brief Allocate memory used by a connected layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    layer_guard_outputs(l);
    layer_guard_weights(l);
    l->state = calloc(l->n_outputs, sizeof(double));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->biases = malloc(sizeof(double) * l->n_outputs);
    l->bias_updates = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->weight_updates = calloc(l->n_weights, sizeof(double));
    l->weight_active = malloc(sizeof(bool) * l->n_weights);
    l->weights = malloc(sizeof(double) * l->n_weights);
    l->mu = malloc(sizeof(double) * N_MU);
}

/**
 * @brief Initialises a fully-connected layer.
 * @param [in] l Layer to initialise.
 * @param [in] args Parameters to initialise the layer.
 */
void
neural_layer_connected_init(struct Layer *l, const struct ArgsLayer *args)
{
    l->options = layer_args_opt(args);
    l->function = args->function;
    l->n_inputs = args->n_inputs;
    l->n_outputs = args->n_init;
    l->max_outputs = args->n_max;
    l->out_w = l->n_outputs;
    l->out_h = 1;
    l->out_c = 1;
    l->n_weights = l->n_inputs * l->n_outputs;
    l->n_biases = l->n_outputs;
    l->n_active = l->n_weights;
    l->eta_max = args->eta;
    l->eta_min = args->eta_min;
    l->momentum = args->momentum;
    l->max_neuron_grow = args->max_neuron_grow;
    l->decay = args->decay;
    layer_init_eta(l);
    malloc_layer_arrays(l);
    for (int i = 0; i < l->n_weights; ++i) {
        l->weights[i] = rand_normal(0, WEIGHT_SD_INIT);
        l->weight_active[i] = true;
    }
    memset(l->biases, 0, sizeof(double) * l->n_biases);
    sam_init(l->mu, N_MU, MU_TYPE);
}

/**
 * @brief Free memory used by a connected layer.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_connected_free(const struct Layer *l)
{
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

/**
 * @brief Initialises and creates a copy of one connected layer from another.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_connected_copy(const struct Layer *src)
{
    if (src->type != CONNECTED) {
        printf("neural_layer_connected_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_defaults(l);
    l->type = src->type;
    l->layer_vptr = src->layer_vptr;
    l->function = src->function;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->out_c = src->out_c;
    l->out_h = src->out_h;
    l->out_w = src->out_w;
    l->n_weights = src->n_weights;
    l->n_biases = src->n_biases;
    l->options = src->options;
    l->eta = src->eta;
    l->eta_max = src->eta_max;
    l->eta_min = src->eta_min;
    l->momentum = src->momentum;
    l->decay = src->decay;
    l->max_neuron_grow = src->max_neuron_grow;
    l->n_active = src->n_active;
    malloc_layer_arrays(l);
    memcpy(l->biases, src->biases, sizeof(double) * src->n_biases);
    memcpy(l->weights, src->weights, sizeof(double) * src->n_weights);
    memcpy(l->weight_active, src->weight_active, sizeof(bool) * src->n_weights);
    memcpy(l->mu, src->mu, sizeof(double) * N_MU);
    return l;
}

/**
 * @brief Randomises a connected layer weights.
 * @param [in] l The layer to randomise.
 */
void
neural_layer_connected_rand(struct Layer *l)
{
    layer_weight_rand(l);
}

/**
 * @brief Forward propagates a connected layer.
 * @param [in] l Layer to forward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input Input to the layer.
 */
void
neural_layer_connected_forward(const struct Layer *l, const struct Net *net,
                               const double *input)
{
    (void) net;
    const int k = l->n_inputs;
    const int n = l->n_outputs;
    const double *a = input;
    const double *b = l->weights;
    double *c = l->state;
    memcpy(l->state, l->biases, sizeof(double) * l->n_outputs);
    blas_gemm(0, 1, 1, n, k, 1, a, k, b, k, 1, c, n);
    neural_activate_array(l->state, l->output, l->n_outputs, l->function);
}

/**
 * @brief Backward propagates a connected layer.
 * @param [in] l The layer to backward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_connected_backward(const struct Layer *l, const struct Net *net,
                                const double *input, double *delta)
{
    (void) net;
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

/**
 * @brief Updates the weights and biases of a connected layer.
 * @param [in] l The layer to update.
 */
void
neural_layer_connected_update(const struct Layer *l)
{
    if (l->options & LAYER_SGD_WEIGHTS && l->eta > 0) {
        blas_axpy(l->n_biases, l->eta, l->bias_updates, 1, l->biases, 1);
        blas_scal(l->n_biases, l->momentum, l->bias_updates, 1);
        if (l->decay > 0) {
            blas_axpy(l->n_weights, -(l->decay), l->weights, 1,
                      l->weight_updates, 1);
        }
        blas_axpy(l->n_weights, l->eta, l->weight_updates, 1, l->weights, 1);
        blas_scal(l->n_weights, l->momentum, l->weight_updates, 1);
        layer_weight_clamp(l);
    }
}

/**
 * @brief Resizes a connected layer if the previous layer has changed size.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_connected_resize(struct Layer *l, const struct Layer *prev)
{
    const int n_weights = prev->n_outputs * l->n_outputs;
    if (n_weights < 1 || n_weights > N_WEIGHTS_MAX) {
        printf("neural_layer_connected: malloc() invalid resize\n");
        layer_print(l, false);
        exit(EXIT_FAILURE);
    }
    double *weights = malloc(sizeof(double) * n_weights);
    double *weight_updates = malloc(sizeof(double) * n_weights);
    bool *weight_active = malloc(sizeof(bool) * n_weights);
    for (int i = 0; i < l->n_outputs; ++i) {
        const int orig_offset = i * l->n_inputs;
        const int offset = i * prev->n_outputs;
        for (int j = 0; j < prev->n_outputs; ++j) {
            if (j < l->n_inputs) {
                weights[offset + j] = l->weights[orig_offset + j];
                weight_updates[offset + j] = l->weight_updates[orig_offset + j];
                weight_active[offset + j] = l->weight_active[orig_offset + j];
            } else {
                weights[offset + j] = rand_normal(0, WEIGHT_SD);
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

/**
 * @brief Mutates a connected layer.
 * @param [in] l The layer to mutate.
 * @return Whether any alterations were made.
 */
bool
neural_layer_connected_mutate(struct Layer *l)
{
    sam_adapt(l->mu, N_MU, MU_TYPE);
    bool mod = false;
    if ((l->options & LAYER_EVOLVE_ETA) && layer_mutate_eta(l, l->mu[0])) {
        mod = true;
    }
    if (l->options & LAYER_EVOLVE_NEURONS) {
        const int n = layer_mutate_neurons(l, l->mu[1]);
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

/**
 * @brief Returns the output from a connected layer.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_connected_output(const struct Layer *l)
{
    return l->output;
}

/**
 * @brief Prints a connected layer.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_connected_print(const struct Layer *l, const bool print_weights)
{
    printf("%s\n", neural_layer_connected_json_export(l, print_weights));
}

/**
 * @brief Returns a json formatted string representation of a connected layer.
 * @param [in] l The layer to return.
 * @param [in] return_weights Whether to returnprint the values of weights and
 * biases.
 * @return String encoded in json format.
 */
const char *
neural_layer_connected_json_export(const struct Layer *l,
                                   const bool return_weights)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "connected");
    cJSON_AddStringToObject(json, "activation",
                            neural_activation_string(l->function));
    cJSON_AddNumberToObject(json, "n_inputs", l->n_inputs);
    cJSON_AddNumberToObject(json, "n_outputs", l->n_outputs);
    cJSON_AddNumberToObject(json, "eta", l->eta);
    cJSON *mutation = cJSON_CreateDoubleArray(l->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    cJSON *weights = cJSON_Parse(layer_weight_json(l, return_weights));
    cJSON_AddItemToObject(json, "weights", weights);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Writes a connected layer to a file.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_connected_save(const struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_biases, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_weights, sizeof(int), 1, fp);
    s += fwrite(&l->options, sizeof(uint32_t), 1, fp);
    s += fwrite(&l->function, sizeof(int), 1, fp);
    s += fwrite(&l->max_neuron_grow, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(&l->eta_max, sizeof(double), 1, fp);
    s += fwrite(&l->eta_min, sizeof(double), 1, fp);
    s += fwrite(&l->momentum, sizeof(double), 1, fp);
    s += fwrite(&l->decay, sizeof(double), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(l->weights, sizeof(double), l->n_weights, fp);
    s += fwrite(l->weight_active, sizeof(bool), l->n_weights, fp);
    s += fwrite(l->biases, sizeof(double), l->n_biases, fp);
    s += fwrite(l->bias_updates, sizeof(double), l->n_biases, fp);
    s += fwrite(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fwrite(l->mu, sizeof(double), N_MU, fp);
    return s;
}

/**
 * @brief Reads a connected layer from a file.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_connected_load(struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_biases, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_weights, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->max_neuron_grow, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    s += fread(&l->eta_max, sizeof(double), 1, fp);
    s += fread(&l->eta_min, sizeof(double), 1, fp);
    s += fread(&l->momentum, sizeof(double), 1, fp);
    s += fread(&l->decay, sizeof(double), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    l->out_w = l->n_outputs;
    l->out_c = 1;
    l->out_h = 1;
    malloc_layer_arrays(l);
    s += fread(l->weights, sizeof(double), l->n_weights, fp);
    s += fread(l->weight_active, sizeof(bool), l->n_weights, fp);
    s += fread(l->biases, sizeof(double), l->n_biases, fp);
    s += fread(l->bias_updates, sizeof(double), l->n_biases, fp);
    s += fread(l->weight_updates, sizeof(double), l->n_weights, fp);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    return s;
}
