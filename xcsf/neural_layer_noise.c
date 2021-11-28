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
 * @date 2016--2021.
 * @brief An implementation of a Gaussian noise adding layer.
 */

#include "neural_layer_noise.h"
#include "neural_activations.h"
#include "utils.h"

/**
 * @brief Allocate memory used by a noise layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    layer_guard_outputs(l);
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
    l->state = calloc(l->n_outputs, sizeof(double));
}

/**
 * @brief Free memory used by a noise layer.
 * @param [in] l The layer to be freed.
 */
static void
free_layer_arrays(const struct Layer *l)
{
    free(l->output);
    free(l->delta);
    free(l->state);
}

/**
 * @brief Initialises a noise layer.
 * @param [in] l Layer to initialise.
 * @param [in] args Parameters to initialise the layer.
 */
void
neural_layer_noise_init(struct Layer *l, const struct ArgsLayer *args)
{
    l->n_inputs = args->n_inputs;
    l->n_outputs = args->n_inputs;
    l->max_outputs = args->n_inputs;
    l->out_w = args->width;
    l->out_h = args->height;
    l->out_c = args->channels;
    l->probability = args->probability;
    l->scale = args->scale;
    malloc_layer_arrays(l);
}

/**
 * @brief Initialises and creates a copy of one noise layer from another.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_noise_copy(const struct Layer *src)
{
    if (src->type != NOISE) {
        printf("neural_layer_noise_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_defaults(l);
    l->type = src->type;
    l->layer_vptr = src->layer_vptr;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->out_w = src->out_w;
    l->out_c = src->out_c;
    l->out_h = src->out_h;
    l->max_outputs = src->max_outputs;
    l->probability = src->probability;
    l->scale = src->scale;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Free memory used by a noise layer.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_noise_free(const struct Layer *l)
{
    free_layer_arrays(l);
}

/**
 * @brief Dummy function since noise layers have no weights.
 * @param [in] l A softmax layer.
 */
void
neural_layer_noise_rand(struct Layer *l)
{
    (void) l;
}

/**
 * @brief Forward propagates a noise layer.
 * @param [in] l Layer to forward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input Input to the layer.
 */
void
neural_layer_noise_forward(const struct Layer *l, const struct Net *net,
                           const double *input)
{
    if (!net->train) {
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

/**
 * @brief Backward propagates a noise layer.
 * @param [in] l The layer to backward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_noise_backward(const struct Layer *l, const struct Net *net,
                            const double *input, double *delta)
{
    (void) net;
    (void) input;
    if (delta) {
        for (int i = 0; i < l->n_inputs; ++i) {
            delta[i] += l->delta[i];
        }
    }
}

/**
 * @brief Dummy function since a noise layer has no weights.
 * @param [in] l A noise layer.
 */
void
neural_layer_noise_update(const struct Layer *l)
{
    (void) l;
}

/**
 * @brief Dummy function since a noise layer cannot be mutated.
 * @param [in] l A noise layer.
 * @return False.
 */
bool
neural_layer_noise_mutate(struct Layer *l)
{
    (void) l;
    return false;
}

/**
 * @brief Resizes a noise layer if the previous layer has changed size.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_noise_resize(struct Layer *l, const struct Layer *prev)
{
    l->n_inputs = prev->n_outputs;
    l->n_outputs = prev->n_outputs;
    l->max_outputs = prev->n_outputs;
    l->out_w = prev->out_w;
    l->out_h = prev->out_h;
    l->out_c = prev->out_c;
    free_layer_arrays(l);
    malloc_layer_arrays(l);
}

/**
 * @brief Returns the output from a noise layer.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_noise_output(const struct Layer *l)
{
    return l->output;
}

/**
 * @brief Prints a noise layer.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_noise_print(const struct Layer *l, const bool print_weights)
{
    printf("%s\n", neural_layer_noise_json_export(l, print_weights));
}

/**
 * @brief Returns a json formatted string representation of a noise layer.
 * @param [in] l The layer to return.
 * @param [in] return_weights Whether to return the values of weights and
 * biases.
 * @return String encoded in json format.
 */
const char *
neural_layer_noise_json_export(const struct Layer *l, const bool return_weights)
{
    (void) return_weights;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "noise");
    cJSON_AddNumberToObject(json, "n_inputs", l->n_inputs);
    cJSON_AddNumberToObject(json, "n_outputs", l->n_outputs);
    cJSON_AddNumberToObject(json, "probability", l->probability);
    cJSON_AddNumberToObject(json, "stdev", l->scale);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Writes a noise layer to a file.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_noise_save(const struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->probability, sizeof(double), 1, fp);
    s += fwrite(&l->scale, sizeof(double), 1, fp);
    s += fwrite(&l->out_w, sizeof(int), 1, fp);
    s += fwrite(&l->out_h, sizeof(int), 1, fp);
    s += fwrite(&l->out_c, sizeof(int), 1, fp);
    return s;
}

/**
 * @brief Reads a noise layer from a file.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_noise_load(struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->probability, sizeof(double), 1, fp);
    s += fread(&l->scale, sizeof(double), 1, fp);
    s += fread(&l->out_w, sizeof(int), 1, fp);
    s += fread(&l->out_h, sizeof(int), 1, fp);
    s += fread(&l->out_c, sizeof(int), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
