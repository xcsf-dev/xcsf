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
 * @file neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2012--2020.
 * @brief An implementation of a multi-layer perceptron neural network.
 */

#include "neural.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_noise.h"
#include "neural_layer_recurrent.h"
#include "neural_layer_softmax.h"

/**
 * @brief Initialises an empty neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to initialise.
 */
void
neural_init(const struct XCSF *xcsf, struct NET *net)
{
    (void) xcsf;
    net->head = NULL;
    net->tail = NULL;
    net->n_layers = 0;
    net->n_inputs = 0;
    net->n_outputs = 0;
    net->output = NULL;
}

/**
 * @brief Inserts a layer into a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network receiving the layer.
 * @param l The layer to insert.
 * @param pos The position in the network to insert the layer.
 */
void
neural_layer_insert(const struct XCSF *xcsf, struct NET *net, struct LAYER *l,
                    const int pos)
{
    (void) xcsf;
    if (net->head == NULL || net->tail == NULL) { // empty list
        net->head = malloc(sizeof(struct LLIST));
        net->head->layer = l;
        net->head->prev = NULL;
        net->head->next = NULL;
        net->tail = net->head;
        net->n_inputs = l->n_inputs;
        net->n_outputs = l->n_outputs;
        net->output = l->output;
    } else { // insert
        struct LLIST *iter = net->tail;
        for (int i = 0; i < pos && iter != NULL; ++i) {
            iter = iter->prev;
        }
        struct LLIST *new = malloc(sizeof(struct LLIST));
        new->layer = l;
        new->prev = iter;
        if (iter == NULL) { // new head
            new->next = net->head;
            net->head->prev = new;
            net->head = new;
            net->n_outputs = l->n_outputs;
            net->output = l->output;
        } else {
            new->next = iter->next;
            iter->next = new;
            if (iter->next == NULL) { // new tail
                net->tail = new;
                net->n_inputs = l->n_inputs;
            } else { // middle
                new->next->prev = new;
            }
        }
    }
    ++(net->n_layers);
}

/**
 * @brief Removes a layer from a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network removing the layer.
 * @param pos The position of the layer in the network to be removed.
 */
void
neural_layer_remove(const struct XCSF *xcsf, struct NET *net, const int pos)
{
    // find the layer
    struct LLIST *iter = net->tail;
    for (int i = 0; i < pos && iter != NULL; ++i) {
        iter = iter->prev;
    }
    if (iter == NULL) {
        printf("neural_layer_remove(): error finding layer to remove\n");
        exit(EXIT_FAILURE);
    } else if (iter->next == NULL && iter->prev == NULL) {
        printf("neural_layer_remove(): attempted to remove the only layer\n");
        exit(EXIT_FAILURE);
    }
    // head
    if (iter->prev == NULL) {
        net->head = iter->next;
        if (iter->next != NULL) {
            iter->next->prev = NULL;
        }
        net->output = net->head->layer->output;
        net->n_outputs = net->head->layer->n_outputs;
    }
    // tail
    if (iter->next == NULL) {
        net->tail = iter->prev;
        if (iter->prev != NULL) {
            iter->prev->next = NULL;
        }
    }
    // middle
    if (iter->prev != NULL && iter->next != NULL) {
        iter->next->prev = iter->prev;
        iter->prev->next = iter->next;
    }
    --(net->n_layers);
    layer_free(xcsf, iter->layer);
    free(iter->layer);
    free(iter);
}

/**
 * @brief Copies a neural network.
 * @param xcsf The XCSF data structure.
 * @param dest The destination neural network.
 * @param src The source neural network.
 */
void
neural_copy(const struct XCSF *xcsf, struct NET *dest, const struct NET *src)
{
    neural_init(xcsf, dest);
    const struct LLIST *iter = src->tail;
    while (iter != NULL) {
        const struct LAYER *f = iter->layer;
        struct LAYER *l = layer_copy(xcsf, f);
        neural_layer_insert(xcsf, dest, l, dest->n_layers);
        iter = iter->prev;
    }
}

/**
 * @brief Frees a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to free.
 */
void
neural_free(const struct XCSF *xcsf, struct NET *net)
{
    struct LLIST *iter = net->tail;
    while (iter != NULL) {
        layer_free(xcsf, iter->layer);
        free(iter->layer);
        net->tail = iter->prev;
        free(iter);
        iter = net->tail;
        --(net->n_layers);
    }
}

/**
 * @brief Randomises the layers within a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to randomise.
 */
void
neural_rand(const struct XCSF *xcsf, const struct NET *net)
{
    const struct LLIST *iter = net->tail;
    while (iter != NULL) {
        layer_rand(xcsf, iter->layer);
        iter = iter->prev;
    }
}

/**
 * @brief Mutates a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to mutate.
 * @return Whether any alterations were made.
 */
_Bool
neural_mutate(const struct XCSF *xcsf, const struct NET *net)
{
    _Bool mod = false;
    _Bool do_resize = false;
    const struct LAYER *prev = NULL;
    const struct LLIST *iter = net->tail;
    while (iter != NULL) {
        const int orig_outputs = iter->layer->n_outputs;
        // if the previous layer has grown or shrunk this layer must be resized
        if (do_resize) {
            layer_resize(xcsf, iter->layer, prev);
            do_resize = false;
        }
        // mutate this layer
        if (layer_mutate(xcsf, iter->layer)) {
            mod = true;
        }
        // check if this layer changed size
        if (iter->layer->n_outputs != orig_outputs) {
            do_resize = true;
        }
        // move to next layer
        prev = iter->layer;
        iter = iter->prev;
    }
    return mod;
}

/**
 * @brief Resizes neural network layers as necessary.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to resize.
 */
void
neural_resize(const struct XCSF *xcsf, const struct NET *net)
{
    const struct LAYER *prev = NULL;
    const struct LLIST *iter = net->tail;
    while (iter != NULL) {
        if (prev != NULL && iter->layer->n_inputs != prev->n_outputs) {
            layer_resize(xcsf, iter->layer, prev);
        }
        prev = iter->layer;
        iter = iter->prev;
    }
}

/**
 * @brief Forward propagates a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to propagate.
 * @param input The input state.
 */
void
neural_propagate(const struct XCSF *xcsf, const struct NET *net,
                 const double *input)
{
    const struct LLIST *iter = net->tail;
    while (iter != NULL) {
        layer_forward(xcsf, iter->layer, input);
        input = layer_output(xcsf, iter->layer);
        iter = iter->prev;
    }
}

/**
 * @brief Performs a gradient descent update on a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to be updated.
 * @param truth The desired network output.
 * @param input The input state.
 */
void
neural_learn(const struct XCSF *xcsf, const struct NET *net,
             const double *truth, const double *input)
{
    /* reset deltas */
    const struct LLIST *iter = net->tail;
    while (iter != NULL) {
        memset(iter->layer->delta, 0, sizeof(double) * iter->layer->n_outputs);
        iter = iter->prev;
    }
    // calculate output layer delta
    const struct LAYER *p = net->head->layer;
    for (int i = 0; i < p->n_outputs; ++i) {
        p->delta[i] = truth[i] - p->output[i];
    }
    /* backward phase */
    iter = net->head;
    while (iter != NULL) {
        const struct LAYER *l = iter->layer;
        if (iter->next == NULL) {
            layer_backward(xcsf, l, input, 0);
        } else {
            const struct LAYER *prev = iter->next->layer;
            layer_backward(xcsf, l, prev->output, prev->delta);
        }
        iter = iter->next;
    }
    /* update phase */
    iter = net->tail;
    while (iter != NULL) {
        layer_update(xcsf, iter->layer);
        iter = iter->prev;
    }
}

/**
 * @brief Returns the output of a specified neuron in the output layer of a
 * neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to output.
 * @param IDX Which neuron in the output layer to return.
 * @return The output of the specified neuron.
 */
double
neural_output(const struct XCSF *xcsf, const struct NET *net, const int IDX)
{
    if (IDX < net->n_outputs) {
        return layer_output(xcsf, net->head->layer)[IDX];
    }
    printf("neural_output(): error (%d) >= (%d)\n", IDX, net->n_outputs);
    exit(EXIT_FAILURE);
}

/**
 * @brief Prints a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to print.
 * @param print_weights Whether to print the weights in each layer.
 */
void
neural_print(const struct XCSF *xcsf, const struct NET *net,
             const _Bool print_weights)
{
    const struct LLIST *iter = net->tail;
    int i = 0;
    while (iter != NULL) {
        printf("layer (%d) ", i);
        layer_print(xcsf, iter->layer, print_weights);
        iter = iter->prev;
        ++i;
    }
}

/**
 * @brief Returns the total number of non-zero weights in a neural network.
 * @param xcsf The XCSF data structure.
 * @param net A neural network.
 * @return The calculated network size.
 */
double
neural_size(const struct XCSF *xcsf, const struct NET *net)
{
    (void) xcsf;
    int size = 0;
    const struct LLIST *iter = net->tail;
    while (iter != NULL) {
        const LAYER *l = iter->layer;
        switch (l->layer_type) {
            case CONNECTED:
            case RECURRENT:
            case LSTM:
            case CONVOLUTIONAL:
                size += l->n_active;
                break;
            default:
                break;
        }
        iter = iter->prev;
    }
    return size;
}

/**
 * @brief Writes a neural network to a binary file.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to save.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_save(const struct XCSF *xcsf, const struct NET *net, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&net->n_layers, sizeof(int), 1, fp);
    s += fwrite(&net->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&net->n_outputs, sizeof(int), 1, fp);
    const struct LLIST *iter = net->tail;
    while (iter != NULL) {
        s += fwrite(&iter->layer->layer_type, sizeof(int), 1, fp);
        s += layer_save(xcsf, iter->layer, fp);
        iter = iter->prev;
    }
    return s;
}

/**
 * @brief Reads a neural network from a binary file.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to load.
 * @param fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_load(const struct XCSF *xcsf, struct NET *net, FILE *fp)
{
    size_t s = 0;
    int nlayers = 0;
    int ninputs = 0;
    int noutputs = 0;
    s += fread(&nlayers, sizeof(int), 1, fp);
    s += fread(&ninputs, sizeof(int), 1, fp);
    s += fread(&noutputs, sizeof(int), 1, fp);
    neural_init(xcsf, net);
    for (int i = 0; i < nlayers; ++i) {
        struct LAYER *l = malloc(sizeof(struct LAYER));
        layer_init(l);
        s += fread(&l->layer_type, sizeof(int), 1, fp);
        layer_set_vptr(l);
        s += layer_load(xcsf, l, fp);
        neural_layer_insert(xcsf, net, l, i);
    }
    return s;
}
