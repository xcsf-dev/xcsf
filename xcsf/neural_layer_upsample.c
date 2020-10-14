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
 * @file neural_layer_upsample.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of a 2D upsampling layer.
 */

#include "neural_layer_upsample.h"
#include "neural_activations.h"
#include "utils.h"
#include "xcsf.h"

/**
 * @brief Allocate memory used by an upsampling layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX) {
        printf("neural_layer_upsample: malloc() invalid size\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
}

/**
 * @brief Initialises a 2D upsampling layer.
 * @param [in] l Layer to initialise.
 * @param [in] args Parameters to initialise the layer.
 */
void
neural_layer_upsample_init(struct Layer *l, const struct LayerArgs *args)
{
    l->height = args->height;
    l->width = args->width;
    l->channels = args->channels;
    l->stride = args->stride;
    l->out_c = args->channels;
    l->out_w = l->width * l->stride;
    l->out_h = l->height * l->stride;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = l->height * l->width * l->channels;
    malloc_layer_arrays(l);
}

/**
 * @brief Initialises and copies one upsampling layer from another.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_upsample_copy(const struct Layer *src)
{
    if (src->layer_type != UPSAMPLE) {
        printf("neural_layer_upsample_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_defaults(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->height = src->height;
    l->width = src->width;
    l->channels = src->channels;
    l->out_w = src->out_w;
    l->out_h = src->out_h;
    l->out_c = src->out_c;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->n_inputs = src->n_inputs;
    l->stride = src->stride;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Free memory used by an upsampling layer.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_upsample_free(const struct Layer *l)
{
    free(l->output);
    free(l->delta);
}

/**
 * @brief Dummy function since upsampling layers have no weights.
 * @param [in] l An average pooling layer.
 */
void
neural_layer_upsample_rand(struct Layer *l)
{
    (void) l;
}

/**
 * @brief Forward propagates an upsampling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to forward propagate.
 * @param [in] input The input to the layer.
 */
void
neural_layer_upsample_forward(const struct XCSF *xcsf, const struct Layer *l,
                              const double *input)
{
    (void) xcsf;
    const int w = l->width;
    const int h = l->height;
    const int c = l->channels;
    const int s = l->stride;
    for (int k = 0; k < c; ++k) {
        for (int j = 0; j < h * s; ++j) {
            for (int i = 0; i < w * s; ++i) {
                const int in_index = k * w * h + (j / s) * w + i / s;
                const int out_index = k * w * h * s * s + j * w * s + i;
                l->output[out_index] = input[in_index];
            }
        }
    }
}

/**
 * @brief Backward propagates an upsampling layer.
 * @param [in] l The layer to backward propagate.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_upsample_backward(const struct Layer *l, const double *input,
                               double *delta)
{
    (void) input;
    if (!delta) {
        return;
    }
    const int w = l->width;
    const int h = l->height;
    const int c = l->channels;
    const int s = l->stride;
    for (int k = 0; k < c; ++k) {
        for (int j = 0; j < h * s; ++j) {
            for (int i = 0; i < w * s; ++i) {
                const int in_index = k * w * h + (j / s) * w + i / s;
                const int out_index = k * w * h * s * s + j * w * s + i;
                delta[in_index] += l->delta[out_index];
            }
        }
    }
}

/**
 * @brief Dummy function since upsampling layers have no weights.
 * @param [in] l An upsampling layer.
 */
void
neural_layer_upsample_update(const struct Layer *l)
{
    (void) l;
}

/**
 * @brief Dummy function since upsampling layers cannot be mutated.
 * @param [in] l An upsampling layer.
 * @return False.
 */
bool
neural_layer_upsample_mutate(struct Layer *l)
{
    (void) l;
    return false;
}

/**
 * @brief Resizes an upsampling layer if the previous layer has changed size.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_upsample_resize(struct Layer *l, const struct Layer *prev)
{
    l->width = prev->out_w;
    l->height = prev->out_h;
    l->channels = prev->out_c;
    l->out_c = prev->out_c;
    l->out_w = l->width * l->stride;
    l->out_h = l->height * l->stride;
    l->n_inputs = l->height * l->width * l->channels;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
}

/**
 * @brief Returns the output from an upsampling layer.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_upsample_output(const struct Layer *l)
{
    return l->output;
}

/**
 * @brief Prints an upsampling layer.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_upsample_print(const struct Layer *l, const bool print_weights)
{
    (void) print_weights;
    printf("upsample in=%d, out=%d, h=%d, w=%d, c=%d, stride=%d\n", l->n_inputs,
           l->n_outputs, l->height, l->width, l->channels, l->stride);
}

/**
 * @brief Writes an upsampling layer to a file.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_upsample_save(const struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&l->height, sizeof(int), 1, fp);
    s += fwrite(&l->width, sizeof(int), 1, fp);
    s += fwrite(&l->channels, sizeof(int), 1, fp);
    s += fwrite(&l->out_w, sizeof(int), 1, fp);
    s += fwrite(&l->out_h, sizeof(int), 1, fp);
    s += fwrite(&l->out_c, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->stride, sizeof(int), 1, fp);
    return s;
}

/**
 * @brief Reads an upsampling layer from a file.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_upsample_load(struct Layer *l, FILE *fp)
{
    size_t s = 0;
    s += fread(&l->height, sizeof(int), 1, fp);
    s += fread(&l->width, sizeof(int), 1, fp);
    s += fread(&l->channels, sizeof(int), 1, fp);
    s += fread(&l->out_w, sizeof(int), 1, fp);
    s += fread(&l->out_h, sizeof(int), 1, fp);
    s += fread(&l->out_c, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->stride, sizeof(int), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
