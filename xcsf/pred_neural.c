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
 * @file pred_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2021.
 * @brief Multi-layer perceptron neural network prediction functions.
 */

#include "pred_neural.h"
#include "neural_activations.h"
#include "neural_layer_avgpool.h"
#include "neural_layer_connected.h"
#include "neural_layer_convolutional.h"
#include "neural_layer_dropout.h"
#include "neural_layer_lstm.h"
#include "neural_layer_maxpool.h"
#include "neural_layer_noise.h"
#include "neural_layer_recurrent.h"
#include "neural_layer_softmax.h"
#include "neural_layer_upsample.h"
#include "utils.h"

/**
 * @brief Creates and initialises a neural network prediction.
 * @details Uses fully-connected layers.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be initialised.
 */
void
pred_neural_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct PredNeural *new = malloc(sizeof(struct PredNeural));
    neural_create(&new->net, xcsf->pred->largs);
    c->pred = new;
}

/**
 * @brief Frees the memory used by a neural network prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be freed.
 */
void
pred_neural_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct PredNeural *pred = c->pred;
    neural_free(&pred->net);
    free(pred);
}

/**
 * @brief Copies a neural network prediction from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
void
pred_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (void) xcsf;
    struct PredNeural *new = malloc(sizeof(struct PredNeural));
    const struct PredNeural *src_pred = src->pred;
    neural_copy(&new->net, &src_pred->net);
    dest->pred = new;
}

/**
 * @brief Backward propagates and updates a neural network prediction.
 * @pre The prediction has been forward propagated for the current state.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose prediction is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
pred_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const double *y)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    neural_learn(&pred->net, y, x);
}

/**
 * @brief Forward propagates a neural network prediction with a provided input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier calculating the prediction.
 * @param [in] x The input state.
 */
void
pred_neural_compute(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    struct PredNeural *pred = c->pred;
    neural_propagate(&pred->net, x, xcsf->explore);
    for (int i = 0; i < xcsf->y_dim; ++i) {
        c->prediction[i] = neural_output(&pred->net, i);
    }
}

/**
 * @brief Prints a neural network prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be printed.
 */
void
pred_neural_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", pred_neural_json_export(xcsf, c));
}

/**
 * @brief Dummy function since neural predictions do not perform crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose prediction is being crossed.
 * @param [in] c2 The second classifier whose prediction is being crossed.
 * @return False.
 */
bool
pred_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Mutates a neural network prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is being mutated.
 * @return Whether any alterations were made.
 */
bool
pred_neural_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    return neural_mutate(&pred->net);
}

/**
 * @brief Returns the size of a neural network prediction.
 * @see neural_size()
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction size to return.
 * @return The network size.
 */
double
pred_neural_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    return neural_size(&pred->net);
}

/**
 * @brief Writes a neural network prediction to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
pred_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    size_t s = neural_save(&pred->net, fp);
    return s;
}

/**
 * @brief Reads a neural network prediction from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
pred_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    struct PredNeural *new = malloc(sizeof(struct PredNeural));
    size_t s = neural_load(&new->net, fp);
    c->pred = new;
    return s;
}

/**
 * @brief Returns the gradient descent rate of a neural prediction layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier maintaining a neural network prediction.
 * @param [in] layer Position of a layer in the network.
 * @return The current gradient descent rate of a layer.
 */
double
pred_neural_eta(const struct XCSF *xcsf, const struct Cl *c, const int layer)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    const struct Llist *iter = pred->net.tail;
    int i = 0;
    while (iter != NULL) {
        if (i == layer) {
            return iter->layer->eta;
        }
        iter = iter->prev;
        ++i;
    }
    return 0;
}

/**
 * @brief Returns the number of neurons in a neural prediction layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier maintaining a neural network prediction.
 * @param [in] layer Position of a layer in the network.
 * @return The current number of neurons in a layer.
 */
int
pred_neural_neurons(const struct XCSF *xcsf, const struct Cl *c,
                    const int layer)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    const struct Llist *iter = pred->net.tail;
    int i = 0;
    while (iter != NULL) {
        if (i == layer) {
            if (iter->layer->type == CONVOLUTIONAL) {
                return iter->layer->n_filters;
            }
            return iter->layer->n_outputs;
        }
        iter = iter->prev;
        ++i;
    }
    return 0;
}

/**
 * @brief Returns the number of active connections in a neural prediction layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier maintaining a neural network prediction.
 * @param [in] layer Position of a layer in the network.
 * @return The current number of active (non-zero) connections in a layer.
 */
int
pred_neural_connections(const struct XCSF *xcsf, const struct Cl *c,
                        const int layer)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    const struct Llist *iter = pred->net.tail;
    int i = 0;
    while (iter != NULL) {
        if (i == layer) {
            return iter->layer->n_active;
        }
        iter = iter->prev;
        ++i;
    }
    return 0;
}

/**
 * @brief Returns the number of layers within a neural network prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier maintaining a neural network prediction.
 * @return The number of layers.
 */
int
pred_neural_layers(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    const struct Net *net = &pred->net;
    return net->n_layers;
}

/**
 * @brief Creates and inserts a hidden layer before the prediction output layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose neural prediction is to be expanded.
 */
void
pred_neural_expand(const struct XCSF *xcsf, const struct Cl *c)
{
    struct PredNeural *pred = c->pred;
    struct Net *net = &pred->net;
    const struct Layer *h = NULL;
    int n_inputs = 0;
    if (net->n_layers > 1) { // select top hidden layer
        h = net->head->next->layer;
        n_inputs = h->n_outputs;
    } else { // if only one layer, must use output layer
        h = net->head->layer;
        n_inputs = h->n_inputs;
    }
    const struct ArgsLayer *largs = xcsf->pred->largs;
    struct ArgsLayer new;
    layer_args_init(&new);
    new.type = CONNECTED;
    new.function = largs->function;
    new.n_inputs = n_inputs;
    new.n_init = h->n_outputs;
    new.n_max = h->max_outputs;
    new.evolve_connect = largs->evolve_connect;
    new.evolve_weights = largs->evolve_weights;
    new.evolve_functions = largs->evolve_functions;
    new.evolve_eta = largs->evolve_eta;
    new.sgd_weights = largs->sgd_weights;
    new.eta = largs->eta;
    new.eta_min = largs->eta_min;
    new.momentum = largs->momentum;
    new.decay = largs->decay;
    struct Layer *l = layer_init(&new);
    const int pos = net->n_layers - 1;
    neural_insert(net, l, pos);
    neural_resize(net);
}

/**
 * @brief Removes prediction (decoder) layers and inserts softmax output layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to convert from autoencoding to classification.
 * @param [in] n_del Number of (decoder) layers to remove from the network.
 */
void
pred_neural_ae_to_classifier(const struct XCSF *xcsf, const struct Cl *c,
                             const int n_del)
{
    struct PredNeural *pred = c->pred;
    struct Net *net = &pred->net;
    struct Layer *l = NULL;
    // remove decoder layers
    for (int i = 0; i < n_del && net->n_layers > 1; ++i) {
        neural_pop(net);
    }
    // add new softmax output
    const struct ArgsLayer *largs = xcsf->pred->largs;
    struct ArgsLayer new;
    layer_args_init(&new);
    new.type = CONNECTED;
    new.function = LINEAR;
    new.n_inputs = net->n_outputs;
    new.n_init = xcsf->y_dim;
    new.n_max = xcsf->y_dim;
    new.evolve_connect = largs->evolve_connect;
    new.evolve_weights = largs->evolve_weights;
    new.evolve_eta = largs->evolve_eta;
    new.sgd_weights = largs->sgd_weights;
    new.eta = largs->eta;
    new.eta_min = largs->eta_min;
    new.momentum = largs->momentum;
    new.decay = largs->decay;
    l = layer_init(&new);
    neural_push(net, l);
    new.type = SOFTMAX;
    new.n_inputs = xcsf->y_dim;
    new.scale = 1;
    l = layer_init(&new);
    neural_push(net, l);
}

/**
 * @brief Returns a json formatted string representation of a neural prediction.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose prediction is to be returned.
 * @return String encoded in json format.
 */
const char *
pred_neural_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "neural");
    cJSON *network = cJSON_Parse(neural_json_export(&pred->net, false));
    cJSON_AddItemToObject(json, "network", network);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
