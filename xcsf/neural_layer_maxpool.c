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
 * @date 2016--2020.
 * @brief An implementation of a 2D maxpooling layer.
 */

#include "neural_layer_maxpool.h"
#include "neural_activations.h"
#include "utils.h"
#include "xcsf.h"

/**
 * @brief Allocate memory used by a maxpooling layer.
 * @param [in] l The layer to be allocated memory.
 */
static void
malloc_layer_arrays(struct Layer *l)
{
    if (l->n_outputs < 1 || l->n_outputs > N_OUTPUTS_MAX) {
        printf("neural_layer_maxpool: malloc() invalid size\n");
        l->n_outputs = 1;
        exit(EXIT_FAILURE);
    }
    l->indexes = calloc(l->n_outputs, sizeof(int));
    l->output = calloc(l->n_outputs, sizeof(double));
    l->delta = calloc(l->n_outputs, sizeof(double));
}

/**
 * @brief Creates and initialises a 2D maxpooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] h The input height.
 * @param [in] w The input width.
 * @param [in] c The number of input channels.
 * @param [in] size The size of the pooling window.
 * @param [in] stride The strides of the pooling operation.
 * @param [in] pad The padding of the pooling operation.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_maxpool_init(const struct XCSF *xcsf, const int h, const int w,
                          const int c, const int size, const int stride,
                          const int pad)
{
    (void) xcsf;
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = MAXPOOL;
    l->layer_vptr = &layer_maxpool_vtbl;
    l->height = h;
    l->width = w;
    l->channels = c;
    l->pad = pad;
    l->out_w = (w + pad - size) / stride + 1;
    l->out_h = (h + pad - size) / stride + 1;
    l->out_c = c;
    l->n_outputs = l->out_h * l->out_w * l->out_c;
    l->max_outputs = l->n_outputs;
    l->n_inputs = h * w * c;
    l->size = size;
    l->stride = stride;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Initialises and creates a copy of one maxpooling layer from another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_maxpool_copy(const struct XCSF *xcsf, const struct Layer *src)
{
    (void) xcsf;
    if (src->layer_type != MAXPOOL) {
        printf("neural_layer_maxpool_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->height = src->height;
    l->width = src->width;
    l->channels = src->channels;
    l->pad = src->pad;
    l->out_w = src->out_w;
    l->out_h = src->out_h;
    l->out_c = src->out_c;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->n_inputs = src->n_inputs;
    l->size = src->size;
    l->stride = src->stride;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Free memory used by a maxpooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_maxpool_free(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    free(l->indexes);
    free(l->output);
    free(l->delta);
}

/**
 * @brief Dummy function since maxpooling layers have no weights.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l A maxpooling layer.
 */
void
neural_layer_maxpool_rand(const struct XCSF *xcsf, struct Layer *l)
{
    (void) xcsf;
    (void) l;
}

/**
 * @brief Returns the index of the maximum value corresponding to a specified
 * output height, width, and channel for a maxpooling layer.
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
            int cur_h = h_offset + i * l->stride + n;
            int cur_w = w_offset + j * l->stride + m;
            int index = cur_w + l->width * (cur_h + l->height * k);
            if (cur_h >= 0 && cur_h < l->height && cur_w >= 0 &&
                cur_w < l->width && input[index] > max) {
                max_index = index;
                max = input[index];
            }
        }
    }
    return max_index;
}

/**
 * @brief Forward propagates a maxpooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to forward propagate.
 * @param [in] input The input to the layer.
 */
void
neural_layer_maxpool_forward(const struct XCSF *xcsf, const struct Layer *l,
                             const double *input)
{
    (void) xcsf;
    for (int k = 0; k < l->channels; ++k) {
        for (int i = 0; i < l->out_h; ++i) {
            for (int j = 0; j < l->out_w; ++j) {
                const int out_index = j + l->out_w * (i + l->out_h * k);
                const int max_index = max_pool(l, input, i, j, k);
                l->indexes[out_index] = max_index;
                l->output[out_index] = input[max_index];
            }
        }
    }
}

/**
 * @brief Backward propagates a maxpooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to backward propagate.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_maxpool_backward(const struct XCSF *xcsf, const struct Layer *l,
                              const double *input, double *delta)
{
    (void) xcsf;
    (void) input;
    if (delta) {
        for (int i = 0; i < l->n_outputs; ++i) {
            delta[l->indexes[i]] += l->delta[i];
        }
    }
}

/**
 * @brief Dummy function since a maxpooling layer has no weights.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l A maxpooling layer.
 */
void
neural_layer_maxpool_update(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    (void) l;
}

/**
 * @brief Dummy function since a maxpooling layer cannot be mutated.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l A maxpooling layer.
 * @return False.
 */
bool
neural_layer_maxpool_mutate(const struct XCSF *xcsf, struct Layer *l)
{
    (void) xcsf;
    (void) l;
    return false;
}

/**
 * @brief Resizes a maxpooling layer if the previous layer has changed size.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_maxpool_resize(const struct XCSF *xcsf, struct Layer *l,
                            const struct Layer *prev)
{
    (void) xcsf;
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
    l->indexes = realloc(l->indexes, sizeof(int) * l->n_outputs);
    l->output = realloc(l->output, sizeof(double) * l->n_outputs);
    l->delta = realloc(l->delta, sizeof(double) * l->n_outputs);
}

/**
 * @brief Returns the output from a maxpooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_maxpool_output(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    return l->output;
}

/**
 * @brief Prints a maxpooling layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_maxpool_print(const struct XCSF *xcsf, const struct Layer *l,
                           const bool print_weights)
{
    (void) xcsf;
    (void) print_weights;
    printf(
        "maxpool in=%d, out=%d, h=%d, w=%d, c=%d, size=%d, stride=%d, pad=%d\n",
        l->n_inputs, l->n_outputs, l->height, l->width, l->channels, l->size,
        l->stride, l->pad);
}

/**
 * @brief Writes a maxpooling layer to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_maxpool_save(const struct XCSF *xcsf, const struct Layer *l,
                          FILE *fp)
{
    (void) xcsf;
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
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_maxpool_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp)
{
    (void) xcsf;
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
