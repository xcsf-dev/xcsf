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
 * @file neural_layer.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Interface for neural network layers.
 */

#pragma once

#include "xcsf.h"

#define CONNECTED (0) //!< Layer type connected
#define DROPOUT (1) //!< Layer type dropout
#define NOISE (2) //!< Layer type noise
#define SOFTMAX (3) //!< Layer type softmax
#define RECURRENT (4) //!< Layer type recurrent
#define LSTM (5) //!< Layer type LSTM
#define MAXPOOL (6) //!< Layer type maxpooling
#define CONVOLUTIONAL (7) //!< Layer type convolutional
#define AVGPOOL (8) //!< Layer type average pooling
#define UPSAMPLE (9) //!< Layer type upsample

#define LAYER_EVOLVE_WEIGHTS (1 << 0) //!< Layer may evolve weights
#define LAYER_EVOLVE_NEURONS (1 << 1) //!< Layer may evolve neurons
#define LAYER_EVOLVE_FUNCTIONS (1 << 2) //!< Layer may evolve functions
#define LAYER_SGD_WEIGHTS (1 << 3) //!< Layer may perform gradient descent
#define LAYER_EVOLVE_ETA (1 << 4) //!< Layer may evolve rate of gradient descent
#define LAYER_EVOLVE_CONNECT (1 << 5) //!< Layer may evolve connectivity

#define ETA_MIN (0.000001) //!< Minimum gradient descent rate
#define NEURON_MIN (-1000) //!< Minimum neuron state
#define NEURON_MAX (1000) //!< Maximum neuron state
#define WEIGHT_MIN (-10) //!< Minimum value of a weight or bias
#define WEIGHT_MAX (10) //!< Maximum value of a weight or bias
#define N_WEIGHTS_MAX (20000000) //!< Maximum number of weights per layer
#define N_INPUTS_MAX (2000000) //!< Maximum number of inputs per layer
#define N_OUTPUTS_MAX (2000000) //!< Maximum number of outputs per layer

/**
 * @brief Neural network layer data structure.
 */
struct Layer {
    int layer_type; //!< Layer type: CONNECTED, DROPOUT, etc.
    double *state; //!< Current neuron states (before activation function)
    double *output; //!< Current neuron outputs (after activation function)
    uint32_t options; //!< Bitwise layer options permitting evolution, SGD, etc.
    double *weights; //!< Weights for calculating neuron states
    bool *weight_active; //!< Whether each connection is present in the layer
    double *biases; //!< Biases for calculating neuron states
    double *bias_updates; //!< Updates to biases
    double *weight_updates; //!< Updates to weights
    double *delta; //!< Delta for updating weights
    double *mu; //!< Mutation rates
    double eta; //!< Gradient descent rate
    int n_inputs; //!< Number of layer inputs
    int n_outputs; //!< Number of layer outputs
    int max_outputs; //!< Maximum number of neurons in the layer
    int n_weights; //!< Number of layer weights
    int n_biases; //!< Number of layer biases
    int n_active; //!< Number of active weights / connections
    int function; //!< Layer activation function
    double scale; //!< Usage depends on layer implementation
    double probability; //!< Usage depends on layer implementation
    struct LayerVtbl const *layer_vptr; //!< Functions acting on layers
    double *prev_state; //!< Previous state for recursive layers
    struct Layer *input_layer; //!< Recursive layer input
    struct Layer *self_layer; //!< Recursive layer self
    struct Layer *output_layer; //!< Recursive layer output
    int recurrent_function; //!< LSTM
    struct Layer *uf; //!< LSTM
    struct Layer *ui; //!< LSTM
    struct Layer *ug; //!< LSTM
    struct Layer *uo; //!< LSTM
    struct Layer *wf; //!< LSTM
    struct Layer *wi; //!< LSTM
    struct Layer *wg; //!< LSTM
    struct Layer *wo; //!< LSTM
    double *cell; //!< LSTM
    double *prev_cell; //!< LSTM
    double *f; //!< LSTM
    double *i; //!< LSTM
    double *g; //!< LSTM
    double *o; //!< LSTM
    double *c; //!< LSTM
    double *h; //!< LSTM
    double *temp; //!< LSTM
    double *temp2; //!< LSTM
    double *temp3; //!< LSTM
    double *dc; //!< LSTM
    int height; //!< Pool, Conv, and Upsample
    int width; //!< Pool, Conv, and Upsample
    int channels; //!< Pool, Conv, and Upsample
    int pad; //!< Pool and Conv
    int out_w; //!< Pool, Conv, and Upsample
    int out_h; //!< Pool, Conv, and Upsample
    int out_c; //!< Pool, Conv, and Upsample
    int size; //!< Pool and Conv
    int stride; //!< Pool, Conv, and Upsample
    int *indexes; //!< Pool
    int n_filters; //!< Conv
    size_t workspace_size; //!< Conv
};

/**
 * @brief Neural network layer interface data structure.
 * @details Neural network layer implementations must implement these functions.
 */
struct LayerVtbl {
    bool (*layer_impl_mutate)(const struct XCSF *xcsf, struct Layer *l);
    void (*layer_impl_resize)(const struct XCSF *xcsf, struct Layer *l,
                              const struct Layer *prev);
    struct Layer *(*layer_impl_copy)(const struct XCSF *xcsf,
                                     const struct Layer *src);
    void (*layer_impl_free)(const struct XCSF *xcsf, const struct Layer *l);
    void (*layer_impl_rand)(const struct XCSF *xcsf, struct Layer *l);
    void (*layer_impl_print)(const struct XCSF *xcsf, const struct Layer *l,
                             const bool print_weights);
    void (*layer_impl_update)(const struct XCSF *xcsf, const struct Layer *l);
    void (*layer_impl_backward)(const struct XCSF *xcsf, const struct Layer *l,
                                const double *input, double *delta);
    void (*layer_impl_forward)(const struct XCSF *xcsf, const struct Layer *l,
                               const double *input);
    double *(*layer_impl_output)(const struct XCSF *xcsf,
                                 const struct Layer *l);
    size_t (*layer_impl_save)(const struct XCSF *xcsf, const struct Layer *l,
                              FILE *fp);
    size_t (*layer_impl_load)(const struct XCSF *xcsf, struct Layer *l,
                              FILE *fp);
};

/**
 * @brief Writes the layer to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
static inline size_t
layer_save(const struct XCSF *xcsf, const struct Layer *l, FILE *fp)
{
    return (*l->layer_vptr->layer_impl_save)(xcsf, l, fp);
}

/**
 * @brief Reads the layer from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
static inline size_t
layer_load(const struct XCSF *xcsf, struct Layer *l, FILE *fp)
{
    return (*l->layer_vptr->layer_impl_load)(xcsf, l, fp);
}

/**
 * @brief Returns the outputs of a layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer whose outputs are to be returned.
 * @return The layer outputs.
 */
static inline double *
layer_output(const struct XCSF *xcsf, const struct Layer *l)
{
    return (*l->layer_vptr->layer_impl_output)(xcsf, l);
}

/**
 * @brief Forward propagates an input through the layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be forward propagated.
 * @param [in] input The input to the layer.
 */
static inline void
layer_forward(const struct XCSF *xcsf, const struct Layer *l,
              const double *input)
{
    (*l->layer_vptr->layer_impl_forward)(xcsf, l, input);
}

/**
 * @brief Backward propagates the error through a layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be backward propagated.
 * @param [in] input The input to the layer.
 * @param [out] delta The previous layer's delta.
 */
static inline void
layer_backward(const struct XCSF *xcsf, const struct Layer *l,
               const double *input, double *delta)
{
    (*l->layer_vptr->layer_impl_backward)(xcsf, l, input, delta);
}

/**
 * @brief Updates the weights and biases of a layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be updated.
 */
static inline void
layer_update(const struct XCSF *xcsf, const struct Layer *l)
{
    (*l->layer_vptr->layer_impl_update)(xcsf, l);
}

/**
 * @brief Performs layer mutation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to mutate.
 * @return Whether any alterations were made.
 */
static inline bool
layer_mutate(const struct XCSF *xcsf, struct Layer *l)
{
    return (*l->layer_vptr->layer_impl_mutate)(xcsf, l);
}

/**
 * @brief Resizes a layer using the previous layer's inputs
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to mutate.
 * @param [in] prev The layer prior to the one being mutated.
 * @return Whether any alterations were made.
 */
static inline void
layer_resize(const struct XCSF *xcsf, struct Layer *l, const struct Layer *prev)
{
    (*l->layer_vptr->layer_impl_resize)(xcsf, l, prev);
}

/**
 * @brief Creates and returns a copy of a specified layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] src The source layer.
 * @return A new copied layer.
 */
static inline struct Layer *
layer_copy(const struct XCSF *xcsf, const struct Layer *src)
{
    return (*src->layer_vptr->layer_impl_copy)(xcsf, src);
}

/**
 * @brief Frees the memory used by the layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be freed.
 */
static inline void
layer_free(const struct XCSF *xcsf, const struct Layer *l)
{
    (*l->layer_vptr->layer_impl_free)(xcsf, l);
}

/**
 * @brief Randomises a layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be randomised.
 */
static inline void
layer_rand(const struct XCSF *xcsf, struct Layer *l)
{
    (*l->layer_vptr->layer_impl_rand)(xcsf, l);
}

/**
 * @brief Prints the layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] l The layer to be printed.
 * @param [in] print_weights Whether to print the weights.
 */
static inline void
layer_print(const struct XCSF *xcsf, const struct Layer *l,
            const bool print_weights)
{
    (*l->layer_vptr->layer_impl_print)(xcsf, l, print_weights);
}

bool
layer_mutate_connectivity(struct Layer *l, const double mu_enable,
                          const double mu_disable);

bool
layer_mutate_eta(const struct XCSF *xcsf, struct Layer *l, const double mu);

bool
layer_mutate_functions(struct Layer *l, const double mu);

bool
layer_mutate_weights(struct Layer *l, const double mu);

int
layer_mutate_neurons(const struct XCSF *xcsf, const struct Layer *l,
                     const double mu);

void
layer_add_neurons(struct Layer *l, const int n);

void
layer_calc_n_active(struct Layer *l);

void
layer_init(struct Layer *l);

void
layer_init_eta(const struct XCSF *xcsf, struct Layer *l);

void
layer_set_vptr(struct Layer *l);

void
layer_weight_clamp(const struct Layer *l);

void
layer_weight_print(const struct Layer *l, const bool print_weights);

void
layer_weight_rand(const struct XCSF *xcsf, struct Layer *l);

void
layer_ensure_input_represention(struct Layer *l);
