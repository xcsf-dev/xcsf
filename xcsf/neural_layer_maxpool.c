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
 * @file neural_layer_maxpool.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief An implementation of a 2D maxpooling layer.
 */

#include "neural_layer_maxpool.h"
#include "neural_activations.h"
#include "utils.h"
#include <float.h>

/**
 * @brief Allocate memory used by a maxpooling layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    layer_guard_outputs(l);
    l->indexes = calloc(l->n_outputs, sizeof(int));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
}

/**
 * @brief Resize memory used by a maxpooling layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
realloc_layer_arrays(struct Layer *l)
{
    layer_guard_outputs(l);
    l->indexes = realloc(l->indexes, sizeof(int) * l->n_outputs);
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
}

/**
 * @brief Initialises a 2D maxpooling layer.
 * @param [in] l Layer to initialise.
 * @param [in] args Parameters to initialise the layer.
 */
void
neural_layer_maxpool_init(struct Layer *l, const struct ArgsLayer *args)
{
    l->height = args->height;
    l->width = args->width;
    l->channels = args->channels;
    l->pad = args->pad;
    l->size = args->size;
    l->stride = args->stride;
    l->out_w = (l->width + l->pad - l->size) / l->stride + 1;
    l->out_h = (l->height + l->pad - l->size) / l->stride + 1;
    l->out_c = l->channels;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = l->height * l->width * l->channels;
    malloc_layer_arrays(l);
}

/**
 * @brief Initialises and creates a copy of one maxpooling layer from another.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_maxpool_copy(const struct Layer *src)
{
    if (src->type != MAXPOOL) {
        printf("neural_layer_maxpool_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_defaults(l);
    l->type = src->type;
    l->layer_vptr = src->layer_vptr;
    l->height = src->height;
    l->width = src->width;
    l->channels = src->channels;
    l->pad = src->pad;
    l->size = src->size;
    l->stride = src->stride;
    l->out_w = src->out_w;
    l->out_h = src->out_h;
    l->out_c = src->out_c;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->n_inputs = src->n_inputs;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Free memory used by a maxpooling layer.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_maxpool_free(const struct Layer *l)
{
    free(l->indexes);
    free(l->output);
    free(l->delta);
}

/**
 * @brief Dummy function since maxpooling layers have no weights.
 * @param [in] l A maxpooling layer.
 */
void
neural_layer_maxpool_rand(struct Layer *l)
{
    (void) l;
}

/**
 * @brief Returns the index of the maximum input value corresponding to a
 * specified output height, width, and channel for a maxpooling layer.
 * @param [in] l A maxpooling layer.
 * @param [in] input The input to perform a maxpooling operation.
 * @param [in] i Output height index.
 * @param [in] j Output width index.
 * @param [in] k Output channel index.
 * @return The index of the maximum value.
 */
static int
max_pool(const struct Layer *l, const double *input, const int i, const int j,
         const int k)
{
    const int w_offset = -l->pad / 2;
    const int h_offset = w_offset;
    double max = -DBL_MAX;
    int max_index = -1;
    for (int n = 0; n < l->size; ++n) {
        for (int m = 0; m < l->size; ++m) {
            const int cur_h = h_offset + i * l->stride + n;
            const int cur_w = w_offset + j * l->stride + m;
            const int index = cur_w + l->width * (cur_h + l->height * k);
            if (cur_h >= 0 && cur_h < l->height && cur_w >= 0 &&
                cur_w < l->width && index < l->n_inputs && input[index] > max) {
                max_index = index;
                max = input[index];
            }
        }
    }
    if (max_index < 0 || max_index >= l->n_inputs) {
        printf("max_pool() error: invalid max_index: (%d)\n", max_index);
        layer_print(l, false);
        exit(EXIT_FAILURE);
    }
    return max_index;
}

/**
 * @brief Forward propagates a maxpooling layer.
 * @param [in] l Layer to forward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input Input to the layer.
 */
void
neural_layer_maxpool_forward(const struct Layer *l, const struct Net *net,
                             const double *input)
{
    (void) net;
    for (int k = 0; k < l->channels; ++k) {
        for (int i = 0; i < l->out_h; ++i) {
            for (int j = 0; j < l->out_w; ++j) {
                const int out_index = j + l->out_w * (i + l->out_h * k);
                if (out_index < l->n_outputs) {
                    const int max_index = max_pool(l, input, i, j, k);
                    l->indexes[out_index] = max_index;
                    l->output[out_index] = input[max_index];
                }
            }
        }
    }
}

/**
 * @brief Backward propagates a maxpooling layer.
 * @param [in] l The layer to backward propagate.
 * @param [in] net Network containing the layer.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_maxpool_backward(const struct Layer *l, const struct Net *net,
                              const double *input, double *delta)
{
    (void) net;
    (void) input;
    if (delta) {
        for (int i = 0; i < l->n_outputs; ++i) {
            delta[l->indexes[i]] += l->delta[i];
        }
    }
}

/**
 * @brief Dummy function since a maxpooling layer has no weights.
 * @param [in] l A maxpooling layer.
 */
void
neural_layer_maxpool_update(const struct Layer *l)
{
    (void) l;
}

/**
 * @brief Dummy function since a maxpooling layer cannot be mutated.
 * @param [in] l A maxpooling layer.
 * @return False.
 */
bool
neural_layer_maxpool_mutate(struct Layer *l)
{
    (void) l;
    return false;
}

/**
 * @brief Resizes a maxpooling layer if the previous layer has changed size.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_maxpool_resize(struct Layer *l, const struct Layer *prev)
{
    const int w = prev->out_w;
    const int h = prev->out_h;
    const int c = prev->out_c;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->n_inputs = h * w * c;
    l->out_w = (w + l->pad - l->size) / l->stride + 1;
    l->out_h = (h + l->pad - l->size) / l->stride + 1;
    l->out_c = c;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    realloc_layer_arrays(l);
}

/**
 * @brief Returns the output from a maxpooling layer.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_maxpool_output(const struct Layer *l)
{
    return l->output;
}

/**
 * @brief Prints a maxpooling layer.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_maxpool_print(const struct Layer *l, const bool print_weights)
{
    printf("%s\n", neural_layer_maxpool_json(l, print_weights));
}

/**
 * @brief Returns a json formatted string representation of a maxpooling layer.
 * @param [in] l The layer to return.
 * @param [in] return_weights Whether to return the values of weights and
 * biases.
 * @return String encoded in json format.
 */
const char *
neural_layer_maxpool_json(const struct Layer *l, const bool return_weights)
{
    (void) return_weights;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "maxpool");
    cJSON_AddNumberToObject(json, "n_inputs", l->n_inputs);
    cJSON_AddNumberToObject(json, "n_outputs", l->n_outputs);
    cJSON_AddNumberToObject(json, "height", l->height);
    cJSON_AddNumberToObject(json, "width", l->width);
    cJSON_AddNumberToObject(json, "channels", l->channels);
    cJSON_AddNumberToObject(json, "size", l->size);
    cJSON_AddNumberToObject(json, "stride", l->stride);
    cJSON_AddNumberToObject(json, "pad", l->pad);
    cJSON_AddNumberToObject(json, "out_w", l->out_w);
    cJSON_AddNumberToObject(json, "out_h", l->out_h);
    cJSON_AddNumberToObject(json, "out_c", l->out_c);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Writes a maxpooling layer to a file.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_maxpool_save(const struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->height, sizeof(int), 1, fp);
    s += fwrite(&l->width, sizeof(int), 1, fp);
    s += fwrite(&l->channels, sizeof(int), 1, fp);
    s += fwrite(&l->pad, sizeof(int), 1, fp);
    s += fwrite(&l->out_w, sizeof(int), 1, fp);
    s += fwrite(&l->out_h, sizeof(int), 1, fp);
    s += fwrite(&l->out_c, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->size, sizeof(int), 1, fp);
    s += fwrite(&l->stride, sizeof(int), 1, fp);
    return s;
}

/**
 * @brief Reads a maxpooling layer from a file.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_maxpool_load(struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->height, sizeof(int), 1, fp);
    s += fread(&l->width, sizeof(int), 1, fp);
    s += fread(&l->channels, sizeof(int), 1, fp);
    s += fread(&l->pad, sizeof(int), 1, fp);
    s += fread(&l->out_w, sizeof(int), 1, fp);
    s += fread(&l->out_h, sizeof(int), 1, fp);
    s += fread(&l->out_c, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->size, sizeof(int), 1, fp);
    s += fread(&l->stride, sizeof(int), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
