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
 * @date 2012--2021.
 * @brief An implementation of a multi-layer perceptron neural network.
 */

#include "neural.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_noise.h"
#include "neural_layer_recurrent.h"
#include "neural_layer_softmax.h"
#include "utils.h"

/**
 * @brief Initialises an empty neural network.
 * @param [in] net The neural network to initialise.
 */
void
neural_init(struct Net *net)
{
    net->head = NULL;
    net->tail = NULL;
    net->n_layers = 0;
    net->n_inputs = 0;
    net->n_outputs = 0;
    net->output = NULL;
    net->train = false;
}

/**
 * @brief Initialises and creates a new neural network from a parameter list.
 * @param [in] net The neural network to initialise.
 * @param [in] arg List of layer parameters defining the initial network.
 */
void
neural_create(struct Net *net, struct ArgsLayer *arg)
{
    neural_init(net);
    const struct Layer *prev_layer = NULL;
    while (arg != NULL) {
        if (prev_layer != NULL) {
            arg->height = prev_layer->out_h; // pass through n inputs
            arg->width = prev_layer->out_w;
            arg->channels = prev_layer->out_c;
            arg->n_inputs = prev_layer->n_outputs;
            switch (arg->type) {
                case AVGPOOL:
                case MAXPOOL:
                case DROPOUT:
                case UPSAMPLE:
                case SOFTMAX:
                case NOISE:
                    arg->n_init = prev_layer->n_outputs;
                    break;
                default:
                    break;
            }
        }
        struct Layer *l = layer_init(arg);
        neural_push(net, l);
        prev_layer = l;
        arg = arg->next;
    }
    if (net->n_layers < 1 || net->n_outputs < 1 || net->n_inputs < 1) {
        printf("neural_create() error: initialising network\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Inserts a layer into a neural network.
 * @param [in] net The neural network receiving the layer.
 * @param [in] l The layer to insert.
 * @param [in] pos The position in the network to insert the layer.
 */
void
neural_insert(struct Net *net, struct Layer *l, const int pos)
{
    if (net->head == NULL || net->tail == NULL) { // empty list
        net->head = malloc(sizeof(struct Llist));
        net->head->layer = l;
        net->head->prev = NULL;
        net->head->next = NULL;
        net->tail = net->head;
        net->n_inputs = l->n_inputs;
        net->n_outputs = l->n_outputs;
        net->output = l->output;
    } else { // insert
        struct Llist *iter = net->tail;
        for (int i = 0; i < pos && iter != NULL; ++i) {
            iter = iter->prev;
        }
        struct Llist *new = malloc(sizeof(struct Llist));
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
 * @param [in] net The neural network removing the layer.
 * @param [in] pos The position of the layer in the network to be removed.
 */
void
neural_remove(struct Net *net, const int pos)
{
    // find the layer
    struct Llist *iter = net->tail;
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
    layer_free(iter->layer);
    free(iter->layer);
    free(iter);
}

/**
 * @brief Inserts a layer at the head of a neural network.
 * @param [in] net The neural network receiving the layer.
 * @param [in] l The layer to insert.
 */
void
neural_push(struct Net *net, struct Layer *l)
{
    neural_insert(net, l, net->n_layers);
}

/**
 * @brief Removes the layer at the head of a neural network.
 * @param [in] net The neural network receiving the layer.
 */
void
neural_pop(struct Net *net)
{
    neural_remove(net, net->n_layers - 1);
}

/**
 * @brief Copies a neural network.
 * @param [in] dest The destination neural network.
 * @param [in] src The source neural network.
 */
void
neural_copy(struct Net *dest, const struct Net *src)
{
    neural_init(dest);
    const struct Llist *iter = src->tail;
    while (iter != NULL) {
        struct Layer *l = layer_copy(iter->layer);
        neural_push(dest, l);
        iter = iter->prev;
    }
}

/**
 * @brief Frees a neural network.
 * @param [in] net The neural network to free.
 */
void
neural_free(struct Net *net)
{
    struct Llist *iter = net->tail;
    while (iter != NULL) {
        layer_free(iter->layer);
        free(iter->layer);
        net->tail = iter->prev;
        free(iter);
        iter = net->tail;
        --(net->n_layers);
    }
}

/**
 * @brief Randomises the layers within a neural network.
 * @param [in] net The neural network to randomise.
 */
void
neural_rand(const struct Net *net)
{
    const struct Llist *iter = net->tail;
    while (iter != NULL) {
        layer_rand(iter->layer);
        iter = iter->prev;
    }
}

/**
 * @brief Mutates a neural network.
 * @param [in] net The neural network to mutate.
 * @return Whether any alterations were made.
 */
bool
neural_mutate(const struct Net *net)
{
    bool mod = false;
    bool do_resize = false;
    const struct Layer *prev = NULL;
    const struct Llist *iter = net->tail;
    while (iter != NULL) {
        const int orig_outputs = iter->layer->n_outputs;
        // if the previous layer has grown or shrunk this layer must be resized
        if (do_resize) {
            layer_resize(iter->layer, prev);
            do_resize = false;
        }
        // mutate this layer
        if (layer_mutate(iter->layer)) {
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
 * @param [in] net The neural network to resize.
 */
void
neural_resize(const struct Net *net)
{
    const struct Layer *prev = NULL;
    const struct Llist *iter = net->tail;
    while (iter != NULL) {
        if (prev != NULL && iter->layer->n_inputs != prev->n_outputs) {
            layer_resize(iter->layer, prev);
        }
        prev = iter->layer;
        iter = iter->prev;
    }
}

/**
 * @brief Forward propagates a neural network.
 * @param [in] net Neural network to propagate.
 * @param [in] input Input state.
 * @param [in] train Whether the network is in training mode.
 */
void
neural_propagate(struct Net *net, const double *input, const bool train)
{
    net->train = train;
    const struct Llist *iter = net->tail;
    while (iter != NULL) {
        layer_forward(iter->layer, net, input);
        input = layer_output(iter->layer);
        iter = iter->prev;
    }
}

/**
 * @brief Performs a gradient descent update on a neural network.
 * @param [in] net The neural network to be updated.
 * @param [in] truth The desired network output.
 * @param [in] input The input state.
 */
void
neural_learn(const struct Net *net, const double *truth, const double *input)
{
    // reset deltas
    const struct Llist *iter = net->tail;
    while (iter != NULL) {
        memset(iter->layer->delta, 0, sizeof(double) * iter->layer->n_outputs);
        iter = iter->prev;
    }
    // calculate output layer delta
    const struct Layer *p = net->head->layer;
    for (int i = 0; i < p->n_outputs; ++i) {
        p->delta[i] = truth[i] - p->output[i];
    }
    // backward phase
    iter = net->head;
    while (iter != NULL) {
        const struct Layer *l = iter->layer;
        if (iter->next == NULL) {
            layer_backward(l, net, input, 0);
        } else {
            const struct Layer *prev = iter->next->layer;
            layer_backward(l, net, prev->output, prev->delta);
        }
        iter = iter->next;
    }
    // update phase
    iter = net->tail;
    while (iter != NULL) {
        layer_update(iter->layer);
        iter = iter->prev;
    }
}

/**
 * @brief Returns the output of a specified neuron in the output layer of a
 * neural network.
 * @param [in] net The neural network to output.
 * @param [in] IDX Which neuron in the output layer to return.
 * @return The output of the specified neuron.
 */
double
neural_output(const struct Net *net, const int IDX)
{
    if (IDX < 0 || IDX >= net->n_outputs) {
        printf("neural_output(): error (%d) >= (%d)\n", IDX, net->n_outputs);
        exit(EXIT_FAILURE);
    }
    return layer_output(net->head->layer)[IDX];
}

/**
 * @brief Returns the outputs from the output layer of a neural network.
 * @param [in] net The neural network to output.
 * @return The neural network outputs.
 */
double *
neural_outputs(const struct Net *net)
{
    return layer_output(net->head->layer);
}

/**
 * @brief Prints a neural network.
 * @param [in] net The neural network to print.
 * @param [in] print_weights Whether to print the weights in each layer.
 */
void
neural_print(const struct Net *net, const bool print_weights)
{
    printf("%s\n", neural_json_export(net, print_weights));
}

/**
 * @brief Returns a json formatted string representation of a neural network.
 * @param [in] net The neural network to return.
 * @param [in] return_weights Whether to return the weights in each layer.
 * @return String encoded in json format.
 */
const char *
neural_json_export(const struct Net *net, const bool return_weights)
{
    cJSON *json = cJSON_CreateObject();
    const struct Llist *iter = net->tail;
    int i = 0;
    char layer_name[256];
    while (iter != NULL) {
        const char *str = layer_json_export(iter->layer, return_weights);
        cJSON *layer = cJSON_Parse(str);
        snprintf(layer_name, 256, "layer_%d", i);
        cJSON_AddItemToObject(json, layer_name, layer);
        iter = iter->prev;
        ++i;
    }
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Returns the total number of non-zero weights in a neural network.
 * @param [in] net A neural network.
 * @return The calculated network size.
 */
double
neural_size(const struct Net *net)
{
    int size = 0;
    const struct Llist *iter = net->tail;
    while (iter != NULL) {
        const struct Layer *l = iter->layer;
        switch (l->type) {
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
 * @brief Writes a neural network to a file.
 * @param [in] net The neural network to save.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
neural_save(const struct Net *net, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&net->n_layers, sizeof(int), 1, fp);
    s += fwrite(&net->n_inputs, sizeof(int), 1, fp);
    s += fwrite(&net->n_outputs, sizeof(int), 1, fp);
    const struct Llist *iter = net->tail;
    while (iter != NULL) {
        s += fwrite(&iter->layer->type, sizeof(int), 1, fp);
        s += layer_save(iter->layer, fp);
        iter = iter->prev;
    }
    return s;
}

/**
 * @brief Reads a neural network from a file.
 * @param [in] net The neural network to load.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
neural_load(struct Net *net, FILE *fp)
{
    size_t s = 0;
    int nlayers = 0;
    int ninputs = 0;
    int noutputs = 0;
    s += fread(&nlayers, sizeof(int), 1, fp);
    s += fread(&ninputs, sizeof(int), 1, fp);
    s += fread(&noutputs, sizeof(int), 1, fp);
    neural_init(net);
    for (int i = 0; i < nlayers; ++i) {
        struct Layer *l = malloc(sizeof(struct Layer));
        layer_defaults(l);
        s += fread(&l->type, sizeof(int), 1, fp);
        layer_set_vptr(l);
        s += layer_load(l, fp);
        neural_push(net, l);
    }
    return s;
}
