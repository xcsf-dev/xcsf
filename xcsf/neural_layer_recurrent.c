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
 * @date 2016--2021.
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

/**
 * @brief Self-adaptation method for mutating a recurrent layer.
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
 * @brief Allocate memory used by a recurrent layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    layer_guard_outputs(l);
    l->state = calloc(l->n_outputs, sizeof(double));
    l->prev_state = calloc(l->n_outputs, sizeof(double));
    l->mu = malloc(sizeof(double) * N_MU);
}

/**
 * @brief Resize memory used by a recurrent layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
realloc_layer_arrays(struct Layer *l)
{
    layer_guard_outputs(l);
    l->state = realloc(l->state, l->n_outputs * sizeof(double));
    l->prev_state = realloc(l->prev_state, l->n_outputs * sizeof(double));
}

/**
 * @brief Free memory used by a recurrent layer.
 * @param [in] l The layer to be freed.
 */
static void
free_layer_arrays(const struct Layer *l)
{
    free(l->state);
    free(l->prev_state);
    free(l->mu);
}

/**
 * @brief Sets the number of active (non-zero) weights in a recurrent layer.
 * @param [in] l The layer to update the number of active weights.
 */
static void
set_layer_n_active(struct Layer *l)
{
    l->n_active = l->input_layer->n_active + l->self_layer->n_active +
        l->output_layer->n_active;
}

/**
 * @brief Sets the total number of weights in a recurrent layer.
 * @param [in] l The layer to update the total number of weights.
 */
static void
set_layer_n_weights(struct Layer *l)
{
    l->n_weights = l->input_layer->n_weights + l->self_layer->n_weights +
        l->output_layer->n_weights;
}

/**
 * @brief Sets the total number of biases in a recurrent layer.
 * @param [in] l The layer to update the total number of biases.
 */
static void
set_layer_n_biases(struct Layer *l)
{
    l->n_biases = l->input_layer->n_biases + l->self_layer->n_biases +
        l->output_layer->n_biases;
}

/**
 * @brief Mutates the gradient descent rate used to update a recurrent layer.
 * @param [in] l The layer whose gradient descent rate is to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_eta(struct Layer *l)
{
    if ((l->options & LAYER_EVOLVE_ETA) && layer_mutate_eta(l, l->mu[0])) {
        l->input_layer->eta = l->eta;
        l->self_layer->eta = l->eta;
        l->output_layer->eta = l->eta;
        return true;
    }
    return false;
}

/**
 * @brief Mutates the number of neurons in a recurrent layer.
 * @param [in] l The layer whose number of neurons is to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_neurons(struct Layer *l)
{
    if (l->options & LAYER_EVOLVE_NEURONS) {
        const int n = layer_mutate_neurons(l->self_layer, l->mu[1]);
        if (n != 0) {
            layer_add_neurons(l->input_layer, n);
            layer_add_neurons(l->self_layer, n);
            layer_add_neurons(l->output_layer, n);
            layer_resize(l->self_layer, l->input_layer);
            layer_resize(l->output_layer, l->input_layer);
            l->n_outputs = l->output_layer->n_outputs;
            l->out_w = l->n_outputs;
            l->out_h = 1;
            l->out_c = 1;
            l->output = l->output_layer->output;
            l->delta = l->output_layer->delta;
            realloc_layer_arrays(l);
            set_layer_n_weights(l);
            set_layer_n_biases(l);
            set_layer_n_active(l);
            return true;
        }
    }
    return false;
}

/**
 * @brief Mutates the number of active weights in a recurrent layer.
 * @param [in] l The layer whose number of active weights is to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_connectivity(struct Layer *l)
{
    bool mod = false;
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

/**
 * @brief Mutates the magnitude of weights and biases in a recurrent layer.
 * @param [in] l The layer whose weights are to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_weights(struct Layer *l)
{
    bool mod = false;
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

/**
 * @brief Mutates the activation function of a recurrent layer.
 * @param [in] l The layer whose activation function is to be mutated.
 * @return Whether any alterations were made.
 */
static bool
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
 * @brief Initialises a recurrent layer.
 * @param [in] l Layer to initialise.
 * @param [in] args Parameters to initialise the layer.
 */
void
neural_layer_recurrent_init(struct Layer *l, const struct ArgsLayer *args)
{
    l->options = layer_args_opt(args);
    l->function = args->function;
    l->n_inputs = args->n_inputs;
    l->n_outputs = args->n_init;
    l->max_outputs = args->n_max;
    l->out_w = l->n_outputs;
    l->out_c = 1;
    l->out_h = 1;
    struct ArgsLayer *cargs = layer_args_copy(args);
    cargs->type = CONNECTED; // recurrent layer is composed of 3 connected
    cargs->function = LINEAR; // input layer and self layer are linear
    l->input_layer = layer_init(cargs);
    cargs->n_inputs = cargs->n_init; // n_init inputs to self and output layers
    l->self_layer = layer_init(cargs);
    cargs->function = args->function; // output activation
    l->output_layer = layer_init(cargs);
    free(cargs);
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
}

/**
 * @brief Initialises and creates a copy of one recurrent layer from another.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_recurrent_copy(const struct Layer *src)
{
    if (src->type != RECURRENT) {
        printf("neural_layer_recurrent_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_defaults(l);
    l->type = src->type;
    l->layer_vptr = src->layer_vptr;
    l->options = src->options;
    l->function = src->function;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->out_w = src->out_w;
    l->out_c = src->out_c;
    l->out_h = src->out_h;
    l->n_active = src->n_active;
    l->eta = src->eta;
    l->input_layer = layer_copy(src->input_layer);
    l->self_layer = layer_copy(src->self_layer);
    l->output_layer = layer_copy(src->output_layer);
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;
    malloc_layer_arrays(l);
    memcpy(l->mu, src->mu, sizeof(double) * N_MU);
    memcpy(l->prev_state, src->prev_state, sizeof(double) * src->n_outputs);
    return l;
}

/**
 * @brief Free memory used by a recurrent layer.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_recurrent_free(const struct Layer *l)
{
    layer_free(l->input_layer);
    layer_free(l->self_layer);
    layer_free(l->output_layer);
    free(l->input_layer);
    free(l->self_layer);
    free(l->output_layer);
    free_layer_arrays(l);
}

/**
 * @brief Randomises a recurrent layer weights.
 * @param [in] l The layer to randomise.
 */
void
neural_layer_recurrent_rand(struct Layer *l)
{
    layer_rand(l->input_layer);
    layer_rand(l->self_layer);
    layer_rand(l->output_layer);
}

/**
 * @brief Forward propagates a recurrent layer.
 * @param [in] l Layer to forward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input Input to the layer.
 */
void
neural_layer_recurrent_forward(const struct Layer *l, const struct Net *net,
                               const double *input)
{
    memcpy(l->prev_state, l->state, sizeof(double) * l->n_outputs);
    layer_forward(l->input_layer, net, input);
    layer_forward(l->self_layer, net, l->output_layer->output);
    memcpy(l->state, l->input_layer->output, sizeof(double) * l->n_outputs);
    blas_axpy(l->n_outputs, 1, l->self_layer->output, 1, l->state, 1);
    layer_forward(l->output_layer, net, l->state);
}

/**
 * @brief Backward propagates a recurrent layer.
 * @param [in] l The layer to backward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_recurrent_backward(const struct Layer *l, const struct Net *net,
                                const double *input, double *delta)
{
    memset(l->input_layer->delta, 0, sizeof(double) * l->n_outputs);
    memset(l->self_layer->delta, 0, sizeof(double) * l->n_outputs);
    layer_backward(l->output_layer, net, l->state, l->self_layer->delta);
    memcpy(l->input_layer->delta, l->self_layer->delta,
           sizeof(double) * l->n_outputs);
    layer_backward(l->self_layer, net, l->prev_state, 0);
    layer_backward(l->input_layer, net, input, delta);
}

/**
 * @brief Updates the weights and biases of a recurrent layer.
 * @param [in] l The layer to update.
 */
void
neural_layer_recurrent_update(const struct Layer *l)
{
    if (l->options & LAYER_SGD_WEIGHTS && l->eta > 0) {
        layer_update(l->input_layer);
        layer_update(l->self_layer);
        layer_update(l->output_layer);
    }
}

/**
 * @brief Resizes a recurrent layer if the previous layer has changed size.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_recurrent_resize(struct Layer *l, const struct Layer *prev)
{
    layer_resize(l->input_layer, prev);
    l->n_inputs = l->input_layer->n_inputs;
    l->n_active = l->input_layer->n_active + l->self_layer->n_active +
        l->output_layer->n_active;
}

/**
 * @brief Returns the output from a recurrent layer.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_recurrent_output(const struct Layer *l)
{
    return l->output;
}

/**
 * @brief Mutates a recurrent layer.
 * @param [in] l The layer to mutate.
 * @return Whether any alterations were made.
 */
bool
neural_layer_recurrent_mutate(struct Layer *l)
{
    sam_adapt(l->mu, N_MU, MU_TYPE);
    bool mod = false;
    mod = mutate_eta(l) ? true : mod;
    mod = mutate_neurons(l) ? true : mod;
    mod = mutate_connectivity(l) ? true : mod;
    mod = mutate_weights(l) ? true : mod;
    mod = mutate_functions(l) ? true : mod;
    return mod;
}

/**
 * @brief Prints a recurrent layer.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_recurrent_print(const struct Layer *l, const bool print_weights)
{
    printf("%s\n", neural_layer_recurrent_json_export(l, print_weights));
}

/**
 * @brief Returns a json formatted string representation of a recurrent layer.
 * @param [in] l The layer to return.
 * @param [in] return_weights Whether to return the values of weights and
 * biases.
 * @return String encoded in json format.
 */
const char *
neural_layer_recurrent_json_export(const struct Layer *l,
                                   const bool return_weights)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "recurrent");
    cJSON_AddStringToObject(json, "activation",
                            neural_activation_string(l->function));
    cJSON_AddNumberToObject(json, "n_inputs", l->n_inputs);
    cJSON_AddNumberToObject(json, "n_outputs", l->n_outputs);
    cJSON_AddNumberToObject(json, "eta", l->eta);
    cJSON *mutation = cJSON_CreateDoubleArray(l->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    cJSON *il = cJSON_Parse(layer_weight_json(l->input_layer, return_weights));
    cJSON_AddItemToObject(json, "input_layer", il);
    cJSON *sl = cJSON_Parse(layer_weight_json(l->self_layer, return_weights));
    cJSON_AddItemToObject(json, "self_layer", sl);
    cJSON *ol = cJSON_Parse(layer_weight_json(l->output_layer, return_weights));
    cJSON_AddItemToObject(json, "output_layer", ol);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Writes a recurrent layer to a file.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_recurrent_save(const struct Layer *l, FILE *fp)
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
    s += layer_save(l->input_layer, fp);
    s += layer_save(l->self_layer, fp);
    s += layer_save(l->output_layer, fp);
    return s;
}

/**
 * @brief Reads a recurrent layer from a file.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_recurrent_load(struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    s += fread(&l->function, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    l->out_w = l->n_outputs;
    l->out_c = 1;
    l->out_h = 1;
    malloc_layer_arrays(l);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    s += fread(l->state, sizeof(double), l->n_outputs, fp);
    s += fread(l->prev_state, sizeof(double), l->n_outputs, fp);
    s += layer_load(l->input_layer, fp);
    s += layer_load(l->self_layer, fp);
    s += layer_load(l->output_layer, fp);
    return s;
}
