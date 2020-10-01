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
 * @file neural_layer_avgpool.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of an average pooling layer.
 */

#include "neural_layer_avgpool.h"
#include "neural_activations.h"
#include "utils.h"
#include "xcsf.h"

/**
 * @brief Allocate memory used by an average pooling layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX) {
        printf("neural_layer_avgpool: malloc() invalid size\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
}

/**
 * @brief Creates and initialises an average pooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] h The input height.
 * @param [in] w The input width.
 * @param [in] c The number of input channels.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_avgpool_init(const struct XCSF *xcsf, const int h, const int w,
                          const int c)
{
    (void) xcsf;
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = AVGPOOL;
    l->layer_vptr = &layer_avgpool_vtbl;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->out_w = 1;
    l->out_h = 1;
    l->out_c = c;
    l->n_outputs = l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = h * w * c;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Initialises and copies one average pooling layer from another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_avgpool_copy(const struct XCSF *xcsf, const struct Layer *src)
{
    (void) xcsf;
    if (src->layer_type != AVGPOOL) {
        printf("neural_layer_avgpool_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
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
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Free memory used by an average pooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_avgpool_free(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    free(l->output);
    free(l->delta);
}

/**
 * @brief Dummy function since average pooling layers have no weights.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l An average pooling layer.
 */
void
neural_layer_avgpool_rand(const struct XCSF *xcsf, struct Layer *l)
{
    (void) xcsf;
    (void) l;
}

/**
 * @brief Forward propagates an average pooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to forward propagate.
 * @param [in] input The input to the layer.
 */
void
neural_layer_avgpool_forward(const struct XCSF *xcsf, const struct Layer *l,
                             const double *input)
{
    (void) xcsf;
    const int n = l->height * l->width;
    for (int k = 0; k < l->channels; ++k) {
        l->output[k] = 0;
        for (int i = 0; i < n; ++i) {
            l->output[k] += input[i + n * k];
        }
        l->output[k] /= n;
    }
}

/**
 * @brief Backward propagates an average pooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to backward propagate.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_avgpool_backward(const struct XCSF *xcsf, const struct Layer *l,
                              const double *input, double *delta)
{
    (void) xcsf;
    (void) input;
    if (delta) {
        const int n = l->height * l->width;
        for (int k = 0; k < l->channels; ++k) {
            for (int i = 0; i < n; ++i) {
                delta[i + n * k] += l->delta[k] / n;
            }
        }
    }
}

/**
 * @brief Dummy function since average pooling layers have no weights.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l An average pooling layer.
 */
void
neural_layer_avgpool_update(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    (void) l;
}

/**
 * @brief Dummy function since average pooling layers cannot be mutated.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l An average pooling layer.
 * @return False.
 */
_Bool
neural_layer_avgpool_mutate(const struct XCSF *xcsf, struct Layer *l)
{
    (void) xcsf;
    (void) l;
    return false;
}

/**
 * @brief Resizes an avg pooling layer if the previous layer has changed size.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_avgpool_resize(const struct XCSF *xcsf, struct Layer *l,
                            const struct Layer *prev)
{
    (void) xcsf;
    const int h = prev->out_h;
    const int w = prev->out_w;
    const int c = prev->out_c;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->out_c = c;
    l->n_outputs = l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = h * w * c;
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
}

/**
 * @brief Returns the output from an average pooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_avgpool_output(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    return l->output;
}

/**
 * @brief Prints an average pooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_avgpool_print(const struct XCSF *xcsf, const struct Layer *l,
                           const _Bool print_weights)
{
    (void) xcsf;
    (void) print_weights;
    printf("avgpool in=%d, out=%d, h=%d, w=%d, c=%d\n", l->n_inputs,
           l->n_outputs, l->height, l->width, l->channels);
}

/**
 * @brief Writes an average pooling layer to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_avgpool_save(const struct XCSF *xcsf, const struct Layer *l,
                          FILE *fp)
{
    (void) xcsf;
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
    return s;
}

/**
 * @brief Reads an average pooling layer from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_avgpool_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp)
{
    (void) xcsf;
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
    malloc_layer_arrays(l);
    return s;
}
