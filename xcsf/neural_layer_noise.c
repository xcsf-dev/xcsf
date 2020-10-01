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
 * @date 2016--2020.
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
    if (l->n_inputs < 1 || l->n_inputs > N_INPUTS_MAX) {
        printf("neural_layer_noise: malloc() invalid size\n");
        l->n_inputs = 1;
        exit(EXIT_FAILURE);
    }
    l->output = calloc(l->n_inputs, sizeof(double));
    l->delta = calloc(l->n_inputs, sizeof(double));
    l->state = calloc(l->n_inputs, sizeof(double));
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
 * @brief Creates and initialises a noise layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n_inputs The number of inputs.
 * @param [in] probability The probability of adding noise to an input.
 * @param [in] std The standard deviation of the Gaussian noise added.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_noise_init(const struct XCSF *xcsf, const int n_inputs,
                        const double probability, const double std)
{
    (void) xcsf;
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = NOISE;
    l->layer_vptr = &layer_noise_vtbl;
    l->n_inputs = n_inputs;
    l->n_outputs = n_inputs;
    l->max_outputs = n_inputs;
    l->probability = probability;
    l->scale = std;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Initialises and creates a copy of one noise layer from another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] src The source layer.
 * @return A pointer to the new layer.
 */
struct Layer *
neural_layer_noise_copy(const struct XCSF *xcsf, const struct Layer *src)
{
    (void) xcsf;
    if (src->layer_type != NOISE) {
        printf("neural_layer_noise_copy(): incorrect source layer type\n");
        exit(EXIT_FAILURE);
    }
    struct Layer *l = malloc(sizeof(struct Layer));
    layer_init(l);
    l->layer_type = src->layer_type;
    l->layer_vptr = src->layer_vptr;
    l->n_inputs = src->n_inputs;
    l->n_outputs = src->n_outputs;
    l->max_outputs = src->max_outputs;
    l->probability = src->probability;
    l->scale = src->scale;
    malloc_layer_arrays(l);
    return l;
}

/**
 * @brief Free memory used by a noise layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be freed.
 */
void
neural_layer_noise_free(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    free_layer_arrays(l);
}

/**
 * @brief Dummy function since noise layers have no weights.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l A softmax layer.
 */
void
neural_layer_noise_rand(const struct XCSF *xcsf, struct Layer *l)
{
    (void) xcsf;
    (void) l;
}

/**
 * @brief Forward propagates a noise layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to forward propagate.
 * @param [in] input The input to the layer.
 */
void
neural_layer_noise_forward(const struct XCSF *xcsf, const struct Layer *l,
                           const double *input)
{
    if (!xcsf->explore) {
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
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to backward propagate.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's error.
 */
void
neural_layer_noise_backward(const struct XCSF *xcsf, const struct Layer *l,
                            const double *input, double *delta)
{
    (void) xcsf;
    (void) input;
    if (delta) {
        for (int i = 0; i < l->n_inputs; ++i) {
            delta[i] += l->delta[i];
        }
    }
}

/**
 * @brief Dummy function since a noise layer has no weights.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l A noise layer.
 */
void
neural_layer_noise_update(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    (void) l;
}

/**
 * @brief Dummy function since a noise layer cannot be mutated.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l A noise layer.
 * @return False.
 */
_Bool
neural_layer_noise_mutate(const struct XCSF *xcsf, struct Layer *l)
{
    (void) xcsf;
    (void) l;
    return false;
}

/**
 * @brief Resizes a noise layer if the previous layer has changed size.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to resize.
 * @param [in] prev The layer previous to the one being resized.
 */
void
neural_layer_noise_resize(const struct XCSF *xcsf, struct Layer *l,
                          const struct Layer *prev)
{
    (void) xcsf;
    l->n_inputs = prev->n_outputs;
    l->n_outputs = prev->n_outputs;
    l->max_outputs = prev->n_outputs;
    free_layer_arrays(l);
    malloc_layer_arrays(l);
}

/**
 * @brief Returns the output from a noise layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer whose output to return.
 * @return The layer output.
 */
double *
neural_layer_noise_output(const struct XCSF *xcsf, const struct Layer *l)
{
    (void) xcsf;
    return l->output;
}

/**
 * @brief Prints a noise layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to print.
 * @param [in] print_weights Whether to print the values of weights and biases.
 */
void
neural_layer_noise_print(const struct XCSF *xcsf, const struct Layer *l,
                         const _Bool print_weights)
{
    (void) xcsf;
    (void) print_weights;
    printf("noise in = %d, out = %d, prob = %f, stdev = %f\n", l->n_inputs,
           l->n_outputs, l->probability, l->scale);
}

/**
 * @brief Writes a noise layer to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_layer_noise_save(const struct XCSF *xcsf, const struct Layer *l,
                        FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fwrite(&l->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&l->n_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->max_outputs, sizeof(int), 1, fp);
    s += fwrite(&l->probability, sizeof(double), 1, fp);
    s += fwrite(&l->scale, sizeof(double), 1, fp);
    return s;
}

/**
 * @brief Reads a noise layer from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_layer_noise_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp)
{
    (void) xcsf;
    size_t s = 0;
    s += fread(&l->n_inputs, sizeof(int), 1, fp);
    s += fread(&l->n_outputs, sizeof(int), 1, fp);
    s += fread(&l->max_outputs, sizeof(int), 1, fp);
    s += fread(&l->probability, sizeof(double), 1, fp);
    s += fread(&l->scale, sizeof(double), 1, fp);
    malloc_layer_arrays(l);
    return s;
}
