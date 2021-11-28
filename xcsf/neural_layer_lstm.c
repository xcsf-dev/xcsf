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
 * @date 2016--2021.
 * @brief An implementation of a long short-term memory layer.
 * @details Stateful, and with a step of 1.
 * Typically the output activation is TANH and recurrent activation LOGISTIC.
 */

#include "neural_layer_lstm.h"
#include "blas.h"
#include "neural_activations.h"
#include "neural_layer_connected.h"
#include "sam.h"
#include "utils.h"

#define N_MU (6) //!< Number of mutation rates applied to an LSTM layer

/**
 * @brief Self-adaptation method for mutating an LSTM layer.
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
 * @brief Sets the total number of weights in an LSTM layer.
 * @param [in] l The layer to update the total number of weights.
 */
static void
set_layer_n_weights(struct Layer *l)
{
    l->n_weights = l->uf->n_weights + l->ui->n_weights + l->ug->n_weights +
        l->uo->n_weights + l->wf->n_weights + l->wi->n_weights +
        l->wg->n_weights + l->wo->n_weights;
}

/**
 * @brief Sets the total number of biases in an LSTM layer.
 * @param [in] l The layer to update the total number of biases.
 */
static void
set_layer_n_biases(struct Layer *l)
{
    l->n_biases = l->uf->n_biases + l->ui->n_biases + l->ug->n_biases +
        l->uo->n_biases + l->wf->n_biases + l->wi->n_biases + l->wg->n_biases +
        l->wo->n_biases;
}

/**
 * @brief Sets the number of active (non-zero) weights in an LSTM layer.
 * @param [in] l The layer to update the number of active weights.
 */
static void
set_layer_n_active(struct Layer *l)
{
    l->n_active = l->uf->n_active + l->ui->n_active + l->ug->n_active +
        l->uo->n_active + l->wf->n_active + l->wi->n_active + l->wg->n_active +
        l->wo->n_active;
}

/**
 * @brief Allocate memory used by an LSTM layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    layer_guard_outputs(l);
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

/**
 * @brief Free memory used by an LSTM layer.
 * @param [in] l The layer to be freed.
 */
static void
free_layer_arrays(const struct Layer *l)
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

/**
 * @brief Sets the gradient descent rate used to update an LSTM layer.
 * @param [in] l The layer whose gradient descent rate is to be set.
 */
static void
set_eta(struct Layer *l)
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

/**
 * @brief Zeros the deltas used to update an LSTM layer.
 * @param [in] l The layer whose deltas are to be reset.
 */
static void
reset_layer_deltas(const struct Layer *l)
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

/**
 * @brief Mutates the gradient descent rate used to update an LSTM layer.
 * @param [in] l The layer whose gradient descent rate is to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_eta(struct Layer *l)
{
    if (layer_mutate_eta(l->uf, l->mu[0])) {
        set_eta(l);
        return true;
    }
    return false;
}

/**
 * @brief Mutates the number of neurons in an LSTM layer.
 * @param [in] l The layer whose number of neurons is to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_neurons(struct Layer *l)
{
    const int n = layer_mutate_neurons(l->uf, l->mu[1]);
    if (n != 0) {
        layer_add_neurons(l->uf, n);
        layer_add_neurons(l->ui, n);
        layer_add_neurons(l->ug, n);
        layer_add_neurons(l->uo, n);
        layer_add_neurons(l->wf, n);
        layer_add_neurons(l->wi, n);
        layer_add_neurons(l->wg, n);
        layer_add_neurons(l->wo, n);
        layer_resize(l->wf, l->uf);
        layer_resize(l->wi, l->uf);
        layer_resize(l->wg, l->uf);
        layer_resize(l->wo, l->uf);
        l->n_outputs = l->uf->n_outputs;
        l->out_w = l->n_outputs;
        l->out_c = 1;
        l->out_h = 1;
        set_layer_n_weights(l);
        set_layer_n_biases(l);
        set_layer_n_active(l);
        free_layer_arrays(l);
        malloc_layer_arrays(l);
        return true;
    }
    return false;
}

/**
 * @brief Mutates the number of active weights in an LSTM layer.
 * @param [in] l The layer whose number of active weights is to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_connectivity(struct Layer *l)
{
    bool mod = false;
    mod = layer_mutate_connectivity(l->uf, l->mu[2], l->mu[3]) ? true : mod;
    mod = layer_mutate_connectivity(l->ui, l->mu[2], l->mu[3]) ? true : mod;
    mod = layer_mutate_connectivity(l->ug, l->mu[2], l->mu[3]) ? true : mod;
    mod = layer_mutate_connectivity(l->uo, l->mu[2], l->mu[3]) ? true : mod;
    mod = layer_mutate_connectivity(l->wf, l->mu[2], l->mu[3]) ? true : mod;
    mod = layer_mutate_connectivity(l->wi, l->mu[2], l->mu[3]) ? true : mod;
    mod = layer_mutate_connectivity(l->wg, l->mu[2], l->mu[3]) ? true : mod;
    mod = layer_mutate_connectivity(l->wo, l->mu[2], l->mu[3]) ? true : mod;
    set_layer_n_active(l);
    return mod;
}

/**
 * @brief Mutates the magnitude of weights and biases in an LSTM layer.
 * @param [in] l The layer whose weights are to be mutated.
 * @return Whether any alterations were made.
 */
static bool
mutate_weights(struct Layer *l)
{
    bool mod = false;
    mod = layer_mutate_weights(l->uf, l->mu[4]) ? true : mod;
    mod = layer_mutate_weights(l->ui, l->mu[4]) ? true : mod;
    mod = layer_mutate_weights(l->ug, l->mu[4]) ? true : mod;
    mod = layer_mutate_weights(l->uo, l->mu[4]) ? true : mod;
    mod = layer_mutate_weights(l->wf, l->mu[4]) ? true : mod;
    mod = layer_mutate_weights(l->wi, l->mu[4]) ? true : mod;
    mod = layer_mutate_weights(l->wg, l->mu[4]) ? true : mod;
    mod = layer_mutate_weights(l->wo, l->mu[4]) ? true : mod;
    return mod;
}

/**
 * @brief Initialises a long short-term memory layer.
 * @param [in] l Layer to initialise.
 * @param [in] args Parameters to initialise the layer.
 */
void
neural_layer_lstm_init(struct Layer *l, const struct ArgsLayer *args)
{
    l->options = layer_args_opt(args);
    l->function = args->function;
    l->recurrent_function = args->recurrent_function;
    l->n_inputs = args->n_inputs;
    l->n_outputs = args->n_init;
    l->max_outputs = args->n_max;
    l->out_w = l->n_outputs;
    l->out_c = 1;
    l->out_h = 1;
    l->eta_max = args->eta;
    l->momentum = args->momentum;
    l->max_neuron_grow = args->max_neuron_grow;
    l->decay = args->decay;
    struct ArgsLayer *cargs = layer_args_copy(args);
    cargs->type = CONNECTED; // lstm is composed of 8 connected layers
    cargs->function = LINEAR;
    l->uf = layer_init(cargs); // input layers
    l->ui = layer_init(cargs);
    l->ug = layer_init(cargs);
    l->uo = layer_init(cargs);
    cargs->n_inputs = cargs->n_init;
    l->wf = layer_init(cargs); // self layers
    l->wi = layer_init(cargs);
    l->wg = layer_init(cargs);
    l->wo = layer_init(cargs);
    free(cargs);
    set_layer_n_biases(l);
    set_layer_n_weights(l);
    set_layer_n_active(l);
    set_eta(l);
    malloc_layer_arrays(l);
    l->mu = malloc(sizeof(double) * N_MU);
    sam_init(l->mu, N_MU, MU_TYPE);
}

/**
 * @brief Initialises and creates a copy of one LSTM layer from another.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_lstm_copy(const struct Layer *src)
{
    if (src->type != LSTM) {
        printf("neural_layer_lstm_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_defaults(l);
    l->type = src->type;
    l->layer_vptr = src->layer_vptr;
    l->function = src->function;
    l->recurrent_function = src->recurrent_function;
    l->options = src->options;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->out_w = src->out_w;
    l->out_h = src->out_h;
    l->out_c = src->out_c;
    l->n_weights = src->n_weights;
    l->n_biases = src->n_biases;
    l->n_active = src->n_active;
    l->eta = src->eta;
    l->eta_max = src->eta_max;
    l->momentum = src->momentum;
    l->decay = src->decay;
    l->max_neuron_grow = src->max_neuron_grow;
    l->max_outputs = src->max_outputs;
    l->uf = layer_copy(src->uf);
    l->ui = layer_copy(src->ui);
    l->ug = layer_copy(src->ug);
    l->uo = layer_copy(src->uo);
    l->wf = layer_copy(src->wf);
    l->wi = layer_copy(src->wi);
    l->wg = layer_copy(src->wg);
    l->wo = layer_copy(src->wo);
    malloc_layer_arrays(l);
    l->mu = malloc(sizeof(double) * N_MU);
    memcpy(l->mu, src->mu, sizeof(double) * N_MU);
    return l;
}

/**
 * @brief Free memory used by an LSTM layer.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_lstm_free(const struct Layer *l)
{
    layer_free(l->uf);
    layer_free(l->ui);
    layer_free(l->ug);
    layer_free(l->uo);
    layer_free(l->wf);
    layer_free(l->wi);
    layer_free(l->wg);
    layer_free(l->wo);
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

/**
 * @brief Randomises an LSTM layer weights.
 * @param [in] l The layer to randomise.
 */
void
neural_layer_lstm_rand(struct Layer *l)
{
    layer_rand(l->uf);
    layer_rand(l->ui);
    layer_rand(l->ug);
    layer_rand(l->uo);
    layer_rand(l->wf);
    layer_rand(l->wi);
    layer_rand(l->wg);
    layer_rand(l->wo);
}

/**
 * @brief Forward propagates an LSTM layer.
 * @param [in] l The layer to forward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input The input to the layer.
 */
void
neural_layer_lstm_forward(const struct Layer *l, const struct Net *net,
                          const double *input)
{
    layer_forward(l->uf, net, input);
    layer_forward(l->ui, net, input);
    layer_forward(l->ug, net, input);
    layer_forward(l->uo, net, input);
    layer_forward(l->wf, net, l->h);
    layer_forward(l->wi, net, l->h);
    layer_forward(l->wg, net, l->h);
    layer_forward(l->wo, net, l->h);
    memcpy(l->f, l->wf->output, sizeof(double) * l->n_outputs);
    blas_axpy(l->n_outputs, 1, l->uf->output, 1, l->f, 1);
    memcpy(l->i, l->wi->output, sizeof(double) * l->n_outputs);
    blas_axpy(l->n_outputs, 1, l->ui->output, 1, l->i, 1);
    memcpy(l->g, l->wg->output, sizeof(double) * l->n_outputs);
    blas_axpy(l->n_outputs, 1, l->ug->output, 1, l->g, 1);
    memcpy(l->o, l->wo->output, sizeof(double) * l->n_outputs);
    blas_axpy(l->n_outputs, 1, l->uo->output, 1, l->o, 1);
    neural_activate_array(l->f, l->f, l->n_outputs, l->recurrent_function);
    neural_activate_array(l->i, l->i, l->n_outputs, l->recurrent_function);
    neural_activate_array(l->g, l->g, l->n_outputs, l->function);
    neural_activate_array(l->o, l->o, l->n_outputs, l->recurrent_function);
    memcpy(l->temp, l->i, sizeof(double) * l->n_outputs);
    blas_mul(l->n_outputs, l->g, 1, l->temp, 1);
    blas_mul(l->n_outputs, l->f, 1, l->c, 1);
    blas_axpy(l->n_outputs, 1, l->temp, 1, l->c, 1);
    memcpy(l->h, l->c, sizeof(double) * l->n_outputs);
    neural_activate_array(l->h, l->h, l->n_outputs, l->function);
    blas_mul(l->n_outputs, l->o, 1, l->h, 1);
    memcpy(l->cell, l->c, sizeof(double) * l->n_outputs);
    memcpy(l->output, l->h, sizeof(double) * l->n_outputs);
}

/**
 * @brief Backward propagates an LSTM layer.
 * @param [in] l The layer to backward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_lstm_backward(const struct Layer *l, const struct Net *net,
                           const double *input, double *delta)
{
    reset_layer_deltas(l);
    memcpy(l->temp3, l->delta, sizeof(double) * l->n_outputs);
    memcpy(l->temp, l->c, sizeof(double) * l->n_outputs);
    neural_activate_array(l->temp, l->temp, l->n_outputs, l->function);
    memcpy(l->temp2, l->temp3, sizeof(double) * l->n_outputs);
    blas_mul(l->n_outputs, l->o, 1, l->temp2, 1);
    neural_gradient_array(l->temp, l->temp2, l->n_outputs, l->function);
    blas_axpy(l->n_outputs, 1, l->dc, 1, l->temp2, 1);
    memcpy(l->temp, l->c, sizeof(double) * l->n_outputs);
    neural_activate_array(l->temp, l->temp, l->n_outputs, l->function);
    blas_mul(l->n_outputs, l->temp3, 1, l->temp, 1);
    neural_gradient_array(l->o, l->temp, l->n_outputs, l->recurrent_function);
    memcpy(l->wo->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->wo, net, l->prev_state, 0);
    memcpy(l->uo->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->uo, net, input, delta);
    memcpy(l->temp, l->temp2, sizeof(double) * l->n_outputs);
    blas_mul(l->n_outputs, l->i, 1, l->temp, 1);
    neural_gradient_array(l->g, l->temp, l->n_outputs, l->function);
    memcpy(l->wg->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->wg, net, l->prev_state, 0);
    memcpy(l->ug->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->ug, net, input, delta);
    memcpy(l->temp, l->temp2, sizeof(double) * l->n_outputs);
    blas_mul(l->n_outputs, l->g, 1, l->temp, 1);
    neural_gradient_array(l->i, l->temp, l->n_outputs, l->recurrent_function);
    memcpy(l->wi->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->wi, net, l->prev_state, 0);
    memcpy(l->ui->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->ui, net, input, delta);
    memcpy(l->temp, l->temp2, sizeof(double) * l->n_outputs);
    blas_mul(l->n_outputs, l->prev_cell, 1, l->temp, 1);
    neural_gradient_array(l->f, l->temp, l->n_outputs, l->recurrent_function);
    memcpy(l->wf->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->wf, net, l->prev_state, 0);
    memcpy(l->uf->delta, l->temp, sizeof(double) * l->n_outputs);
    layer_backward(l->uf, net, input, delta);
    memcpy(l->temp, l->temp2, sizeof(double) * l->n_outputs);
    blas_mul(l->n_outputs, l->f, 1, l->temp, 1);
    memcpy(l->dc, l->temp, sizeof(double) * l->n_outputs);
}

/**
 * @brief Updates the weights and biases of an LSTM layer.
 * @param [in] l The layer to update.
 */
void
neural_layer_lstm_update(const struct Layer *l)
{
    if (l->options & LAYER_SGD_WEIGHTS && l->eta > 0) {
        layer_update(l->wf);
        layer_update(l->wi);
        layer_update(l->wg);
        layer_update(l->wo);
        layer_update(l->uf);
        layer_update(l->ui);
        layer_update(l->ug);
        layer_update(l->uo);
    }
}

/**
 * @brief Resizes an LSTM layer if the previous layer has changed size.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_lstm_resize(struct Layer *l, const struct Layer *prev)
{
    layer_resize(l->uf, prev);
    layer_resize(l->ui, prev);
    layer_resize(l->ug, prev);
    layer_resize(l->uo, prev);
    layer_resize(l->uf, prev);
    l->n_inputs = prev->n_outputs;
    set_layer_n_weights(l);
    set_layer_n_biases(l);
    set_layer_n_active(l);
}

/**
 * @brief Returns the output from an LSTM layer.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_lstm_output(const struct Layer *l)
{
    return l->output;
}

/**
 * @brief Mutates an LSTM layer.
 * @param [in] l The layer to mutate.
 * @return Whether any alterations were made.
 */
bool
neural_layer_lstm_mutate(struct Layer *l)
{
    sam_adapt(l->mu, N_MU, MU_TYPE);
    bool mod = false;
    if ((l->options & LAYER_EVOLVE_ETA) && mutate_eta(l)) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_NEURONS) && mutate_neurons(l)) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_CONNECT) && mutate_connectivity(l)) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_WEIGHTS) && mutate_weights(l)) {
        mod = true;
    }
    if ((l->options & LAYER_EVOLVE_FUNCTIONS) &&
        layer_mutate_functions(l, l->mu[5])) {
        mod = true;
    }
    return mod;
}

/**
 * @brief Prints an LSTM layer.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_lstm_print(const struct Layer *l, const bool print_weights)
{
    printf("%s\n", neural_layer_lstm_json_export(l, print_weights));
}

/**
 * @brief Returns a json formatted string representation of an LSTM layer.
 * @param [in] l The layer to return.
 * @param [in] return_weights Whether to returnprint the values of weights and
 * biases.
 * @return String encoded in json format.
 */
const char *
neural_layer_lstm_json_export(const struct Layer *l, const bool return_weights)
{
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "lstm");
    cJSON_AddStringToObject(json, "activation",
                            neural_activation_string(l->function));
    cJSON_AddStringToObject(json, "recurrent_activation",
                            neural_activation_string(l->recurrent_function));
    cJSON_AddNumberToObject(json, "n_inputs", l->n_inputs);
    cJSON_AddNumberToObject(json, "n_outputs", l->n_outputs);
    cJSON_AddNumberToObject(json, "eta", l->eta);
    cJSON *mutation = cJSON_CreateDoubleArray(l->mu, N_MU);
    cJSON_AddItemToObject(json, "mutation", mutation);
    cJSON *uf = cJSON_Parse(layer_weight_json(l->uf, return_weights));
    cJSON_AddItemToObject(json, "uf_layer", uf);
    cJSON *ui = cJSON_Parse(layer_weight_json(l->ui, return_weights));
    cJSON_AddItemToObject(json, "ui_layer", ui);
    cJSON *ug = cJSON_Parse(layer_weight_json(l->ug, return_weights));
    cJSON_AddItemToObject(json, "ug_layer", ug);
    cJSON *uo = cJSON_Parse(layer_weight_json(l->uo, return_weights));
    cJSON_AddItemToObject(json, "uo_layer", uo);
    cJSON *wf = cJSON_Parse(layer_weight_json(l->wf, return_weights));
    cJSON_AddItemToObject(json, "wf_layer", wf);
    cJSON *wi = cJSON_Parse(layer_weight_json(l->wi, return_weights));
    cJSON_AddItemToObject(json, "wi_layer", wi);
    cJSON *wg = cJSON_Parse(layer_weight_json(l->wg, return_weights));
    cJSON_AddItemToObject(json, "wg_layer", wg);
    cJSON *wo = cJSON_Parse(layer_weight_json(l->wo, return_weights));
    cJSON_AddItemToObject(json, "wo_layer", wo);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Writes an LSTM layer to a file.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_lstm_save(const struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_weights, sizeof(int), 1, fp);
    s += fwrite(&l->n_biases, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_active, sizeof(int), 1, fp);
    s += fwrite(&l->eta, sizeof(double), 1, fp);
    s += fwrite(&l->eta_max, sizeof(double), 1, fp);
    s += fwrite(&l->momentum, sizeof(double), 1, fp);
    s += fwrite(&l->decay, sizeof(double), 1, fp);
    s += fwrite(&l->max_neuron_grow, sizeof(int), 1, fp);
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
    s += layer_save(l->uf, fp);
    s += layer_save(l->ui, fp);
    s += layer_save(l->ug, fp);
    s += layer_save(l->uo, fp);
    s += layer_save(l->wf, fp);
    s += layer_save(l->wi, fp);
    s += layer_save(l->wg, fp);
    s += layer_save(l->wo, fp);
    return s;
}

/**
 * @brief Reads an LSTM layer from a file.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_lstm_load(struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_weights, sizeof(int), 1, fp);
    s += fread(&l->n_biases, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_active, sizeof(int), 1, fp);
    s += fread(&l->eta, sizeof(double), 1, fp);
    s += fread(&l->eta_max, sizeof(double), 1, fp);
    s += fread(&l->momentum, sizeof(double), 1, fp);
    s += fread(&l->decay, sizeof(double), 1, fp);
    s += fread(&l->max_neuron_grow, sizeof(int), 1, fp);
    s += fread(&l->options, sizeof(uint32_t), 1, fp);
    l->out_w = l->n_outputs;
    l->out_c = 1;
    l->out_h = 1;
    malloc_layer_arrays(l);
    l->mu = malloc(sizeof(double) * N_MU);
    s += fread(l->mu, sizeof(double), N_MU, fp);
    s += fread(l->state, sizeof(double), l->n_outputs, fp);
    s += fread(l->prev_state, sizeof(double), l->n_outputs, fp);
    s += fread(l->cell, sizeof(double), l->n_outputs, fp);
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
    s += layer_load(l->uf, fp);
    s += layer_load(l->ui, fp);
    s += layer_load(l->ug, fp);
    s += layer_load(l->uo, fp);
    s += layer_load(l->wf, fp);
    s += layer_load(l->wi, fp);
    s += layer_load(l->wg, fp);
    s += layer_load(l->wo, fp);
    return s;
}
