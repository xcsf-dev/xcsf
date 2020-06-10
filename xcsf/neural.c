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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_noise.h"
#include "neural_layer_softmax.h"

/**
 * @brief Initialises an empty neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to initialise.
 */
void neural_init(const XCSF *xcsf, NET *net)
{
    (void)xcsf;
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
 * @param p The position in the network to insert the layer.
 */
void neural_layer_insert(const XCSF *xcsf, NET *net, LAYER *l, int p)
{
    (void)xcsf;
    // empty list
    if(net->head == NULL || net->tail == NULL) {
        net->head = malloc(sizeof(LLIST));
        net->head->layer = l;
        net->head->prev = NULL;
        net->head->next = NULL;
        net->tail = net->head;
        net->n_inputs = l->n_inputs;
        net->n_outputs = l->n_outputs;
        net->output = l->output;
    }
    // insert
    else {
        LLIST *iter = net->tail;
        for(int i = 0; i < p && iter != NULL; i++) {
            iter = iter->prev;
        }
        LLIST *new = malloc(sizeof(LLIST));
        new->layer = l;
        new->prev = iter;
        // new head
        if(iter == NULL) {
            new->next = net->head;
            net->head->prev = new;
            net->head = new;
            net->n_outputs = l->n_outputs;
            net->output = l->output;
        } else {
            new->next = iter->next;
            iter->next = new;
            // new tail
            if(iter->next == NULL) {
                net->tail = new;
                net->n_inputs = l->n_inputs;
            }
            // middle
            else {
                new->next->prev = new;
            }
        }
    }
    net->n_layers++;
}

/**
 * @brief Removes a layer from a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network removing the layer.
 * @param p The position of the layer in the network to be removed.
 */
void neural_layer_remove(const XCSF *xcsf, NET *net, int p)
{
    // find the layer
    LLIST *iter = net->tail;
    for(int i = 0; i < p && iter != NULL; i++) {
        iter = iter->prev;
    }
    if(iter == NULL) {
        printf("neural_layer_remove(): error finding layer to remove\n");
        exit(EXIT_FAILURE);
    } else if(iter->next == NULL && iter->prev == NULL) {
        printf("neural_layer_remove(): attempted to remove the only layer\n");
        exit(EXIT_FAILURE);
    }
    // head
    if(iter->prev == NULL) {
        net->head = iter->next;
        if(iter->next != NULL) {
            iter->next->prev = NULL;
        }
        net->output = net->head->layer->output;
        net->n_outputs = net->head->layer->n_outputs;
    }
    // tail
    if(iter->next == NULL) {
        net->tail = iter->prev;
        if(iter->prev != NULL) {
            iter->prev->next = NULL;
        }
    }
    // middle
    if(iter->prev != NULL && iter->next != NULL) {
        iter->next->prev = iter->prev;
        iter->prev->next = iter->next;
    }
    net->n_layers--;
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
void neural_copy(const XCSF *xcsf, NET *dest, const NET *src)
{
    neural_init(xcsf, dest);
    int p = 0;
    for(const LLIST *iter = src->tail; iter != NULL; iter = iter->prev) {
        const LAYER *f = iter->layer;
        LAYER *l = layer_copy(xcsf, f);
        neural_layer_insert(xcsf, dest, l, p);
        p++;
    }
}

/**
 * @brief Frees a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to free.
 */
void neural_free(const XCSF *xcsf, NET *net)
{
    LLIST *iter = net->tail;
    while(iter != NULL) {
        layer_free(xcsf, iter->layer);
        free(iter->layer);
        net->tail = iter->prev;
        free(iter);
        iter = net->tail;
        net->n_layers--;
    }
}

/**
 * @brief Randomises the layers within a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to randomise.
 */
void neural_rand(const XCSF *xcsf, const NET *net)
{
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        layer_rand(xcsf, iter->layer);
    }
}

/**
 * @brief Mutates a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to mutate.
 * @return Whether any alterations were made.
 */
_Bool neural_mutate(const XCSF *xcsf, const NET *net)
{
    _Bool mod = false;
    const LAYER *prev = NULL;
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        // previous layer has grown or shrunk: weight vector must be resized
        if(prev != NULL && iter->layer->n_inputs != prev->n_outputs) {
            layer_resize(xcsf, iter->layer, prev);
        }
        // mutate this layer
        if(layer_mutate(xcsf, iter->layer)) {
            mod = true;
        }
        prev = iter->layer;
    }
    return mod;
}

/**
 * @brief Resizes neural network layers as necessary.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to resize.
 */
void neural_resize(const XCSF *xcsf, const NET *net)
{
    const LAYER *prev = NULL;
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        if(prev != NULL && iter->layer->n_inputs != prev->n_outputs) {
            layer_resize(xcsf, iter->layer, prev);
        }
        prev = iter->layer;
    }
}

/**
 * @brief Forward propagates a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to propagate.
 * @param input The input state.
 */
void neural_propagate(const XCSF *xcsf, const NET *net, const double *input)
{
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        layer_forward(xcsf, iter->layer, input);
        input = layer_output(xcsf, iter->layer);
    }
}

/**
 * @brief Performs a gradient descent update on a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to be updated.
 * @param truth The desired network output.
 * @param input The input state.
 */
void neural_learn(const XCSF *xcsf, NET *net, const double *truth, const double *input)
{
    /* reset deltas */
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        memset(iter->layer->delta, 0, iter->layer->n_outputs * sizeof(double));
    }
    // calculate output layer error
    const LAYER *p = net->head->layer;
    for(int i = 0; i < p->n_outputs; i++) {
        p->delta[i] = (truth[i] - p->output[i]);
    }
    /* backward phase */
    for(const LLIST *iter = net->head; iter != NULL; iter = iter->next) {
        const LAYER *l = iter->layer;
        if(iter->next == NULL) {
            net->input = input;
            net->delta = 0;
        } else {
            const LAYER *prev = iter->next->layer;
            net->input = prev->output;
            net->delta = prev->delta;
        }
        layer_backward(xcsf, l, net);
    }
    /* update phase */
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        layer_update(xcsf, iter->layer);
    }
}

/**
 * @brief Gradient descent updates a pair of active autoencoder layers.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to be updated.
 * @param input The input to the network.
 */
void neural_ae(const XCSF *xcsf, NET *net, const double *input)
{
    // select decoder and encoder layers
    int layer = 0;
    const LLIST *iter = net->tail;
    while(iter != NULL && layer < (net->n_layers / 2)) {
        iter = iter->prev;
        layer++;
    }
    if(iter == NULL) {
        printf("neural_ae(): error finding decoder\n");
        exit(EXIT_FAILURE);
    }
    const LAYER *decoder = iter->layer;
    const LAYER *encoder = iter->next->layer;
    // desired output
    if(iter->next->next != NULL) {
        input = iter->next->next->layer->output;
    }
    // calculate decoder delta
    for(int i = 0; i < decoder->n_outputs; i++) {
        decoder->delta[i] = input[i] - decoder->output[i];
    }
    // reset encoder delta
    memset(encoder->delta, 0, encoder->n_outputs * sizeof(double));
    // backward decoder
    net->input = encoder->output;
    net->delta = encoder->delta;
    layer_backward(xcsf, decoder, net);
    // backward encoder
    net->input = input;
    net->delta = 0;
    layer_backward(xcsf, encoder, net);
    // update
    layer_update(xcsf, decoder);
    layer_update(xcsf, encoder);
}

/**
 * @brief Returns the output of a specified neuron in the output layer of a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to output.
 * @param i Which neuron in the output layer to return.
 * @return The output of the specified neuron.
 */
double neural_output(const XCSF *xcsf, const NET *net, int i)
{
    if(i < net->n_outputs) {
        return layer_output(xcsf, net->head->layer)[i];
    }
    printf("neural_output(): requested (%d) in output layer of size (%d)\n", i, net->n_outputs);
    exit(EXIT_FAILURE);
}

/**
 * @brief Prints a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to print.
 * @param print_weights Whether to print the weights in each layer.
 */
void neural_print(const XCSF *xcsf, const NET *net, _Bool print_weights)
{
    int i = 0;
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        printf("layer (%d) ", i);
        layer_print(xcsf, iter->layer, print_weights);
        i++;
    }
}

/**
 * @brief Returns the total number of hidden neurons in a neural network.
 * @param xcsf The XCSF data structure.
 * @param net The neural network to calculate the number of neurons.
 * @return The total number of hidden neurons.
 */
int neural_size(const XCSF *xcsf, const NET *net)
{
    (void)xcsf;
    int size = 0;
    for(const LLIST *iter = net->tail; iter->prev != NULL; iter = iter->prev) {
        if(iter->layer->layer_type == CONNECTED) {
            size += iter->layer->n_outputs;
        }
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
size_t neural_save(const XCSF *xcsf, const NET *net, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&net->n_layers, sizeof(int), 1, fp);
    s += fwrite(&net->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&net->n_outputs, sizeof(int), 1, fp);
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        s += fwrite(&iter->layer->layer_type, sizeof(int), 1, fp);
        s += layer_save(xcsf, iter->layer, fp);
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
size_t neural_load(const XCSF *xcsf, NET *net, FILE *fp)
{
    size_t s = 0;
    int nlayers = 0;
    int ninputs = 0;
    int noutputs = 0;
    s += fread(&nlayers, sizeof(int), 1, fp);
    s += fread(&ninputs, sizeof(int), 1, fp);
    s += fread(&noutputs, sizeof(int), 1, fp);
    neural_init(xcsf, net);
    for(int i = 0; i < nlayers; i++) {
        LAYER *l = malloc(sizeof(LAYER));
        s += fread(&l->layer_type, sizeof(int), 1, fp);
        neural_layer_set_vptr(l);
        s += layer_load(xcsf, l, fp);
        neural_layer_insert(xcsf, net, l, i);
    }
    return s;
}
