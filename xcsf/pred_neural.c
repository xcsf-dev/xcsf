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
 * @date 2016--2020.
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

/**
 * @brief Creates and initialises a neural network prediction.
 * @details Uses fully-connected layers.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is to be initialised.
 */
void
pred_neural_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct PredNeural *new = malloc(sizeof(struct PredNeural));
    neural_init(xcsf, &new->net);
    // hidden layers
    uint32_t lopt = neural_pred_lopt(xcsf);
    struct Layer *l = NULL;
    int n_inputs = xcsf->x_dim;
    for (int i = 0; i < MAX_LAYERS && xcsf->PRED_NUM_NEURONS[i] > 0; ++i) {
        const int hinit = xcsf->PRED_NUM_NEURONS[i];
        int hmax = xcsf->PRED_MAX_NEURONS[i];
        if (hmax < hinit || !xcsf->PRED_EVOLVE_NEURONS) {
            hmax = hinit;
        }
        const int f = xcsf->PRED_HIDDEN_ACTIVATION;
        l = neural_layer_connected_init(xcsf, n_inputs, hinit, hmax, f, lopt);
        neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
        n_inputs = hinit;
    }
    // output layer
    const int f = xcsf->PRED_OUTPUT_ACTIVATION;
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    lopt &= ~LAYER_EVOLVE_FUNCTIONS; // never evolve the output neurons function
    if (f == SOFT_MAX) { // classification
        l = neural_layer_connected_init(xcsf, n_inputs, xcsf->y_dim,
                                        xcsf->y_dim, LINEAR, lopt);
        neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
        l = neural_layer_softmax_init(xcsf, xcsf->y_dim, 1);
        neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
    } else { // regression
        l = neural_layer_connected_init(xcsf, n_inputs, xcsf->y_dim,
                                        xcsf->y_dim, f, lopt);
        neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
    }
    c->pred = new;
}

void
pred_neural_free(const struct XCSF *xcsf, const struct Cl *c)
{
    struct PredNeural *pred = c->pred;
    neural_free(xcsf, &pred->net);
    free(pred);
}

void
pred_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    struct PredNeural *new = malloc(sizeof(struct PredNeural));
    const struct PredNeural *src_pred = src->pred;
    neural_copy(xcsf, &new->net, &src_pred->net);
    dest->pred = new;
}

void
pred_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const double *y)
{
    if (xcsf->PRED_SGD_WEIGHTS) {
        const struct PredNeural *pred = c->pred;
        neural_learn(xcsf, &pred->net, y, x);
    }
}

void
pred_neural_compute(const struct XCSF *xcsf, const struct Cl *c,
                    const double *x)
{
    const struct PredNeural *pred = c->pred;
    neural_propagate(xcsf, &pred->net, x);
    for (int i = 0; i < xcsf->y_dim; ++i) {
        c->prediction[i] = neural_output(xcsf, &pred->net, i);
    }
}

void
pred_neural_print(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct PredNeural *pred = c->pred;
    neural_print(xcsf, &pred->net, false);
}

_Bool
pred_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
pred_neural_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct PredNeural *pred = c->pred;
    return neural_mutate(xcsf, &pred->net);
}

double
pred_neural_size(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct PredNeural *pred = c->pred;
    return neural_size(xcsf, &pred->net);
}

size_t
pred_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    const struct PredNeural *pred = c->pred;
    size_t s = neural_save(xcsf, &pred->net, fp);
    return s;
}

size_t
pred_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    struct PredNeural *new = malloc(sizeof(struct PredNeural));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->pred = new;
    return s;
}

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
            return iter->layer->n_outputs;
        }
        iter = iter->prev;
        ++i;
    }
    return 0;
}

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

int
pred_neural_layers(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct PredNeural *pred = c->pred;
    const struct Net *net = &pred->net;
    return net->n_layers;
}

void
pred_neural_expand(const struct XCSF *xcsf, const struct Cl *c)
{
    struct PredNeural *pred = c->pred;
    struct Net *net = &pred->net;
    const struct Layer *h = NULL;
    int n_inputs = 0;
    // select top hidden layer
    if (net->n_layers > 1) {
        h = net->head->next->layer;
        n_inputs = h->n_outputs;
    }
    // if only one layer, must use output layer
    else {
        h = net->head->layer;
        n_inputs = h->n_inputs;
    }
    const int n_outputs = h->n_outputs;
    const int max_outputs = h->max_outputs;
    const int f = xcsf->PRED_HIDDEN_ACTIVATION;
    const int pos = net->n_layers - 1;
    const uint32_t lopt = neural_pred_lopt(xcsf);
    struct Layer *l = neural_layer_connected_init(xcsf, n_inputs, n_outputs,
                                                  max_outputs, f, lopt);
    neural_layer_insert(xcsf, net, l, pos);
    neural_resize(xcsf, net);
}

void
pred_neural_ae_to_classifier(const struct XCSF *xcsf, const struct Cl *c,
                             const int n_del)
{
    struct PredNeural *pred = c->pred;
    struct Net *net = &pred->net;
    struct Layer *l = NULL;
    // remove decoder layers
    for (int i = 0; i < n_del && net->n_layers > 1; ++i) {
        neural_layer_remove(xcsf, net, net->n_layers - 1);
    }
    // add new softmax output
    const int code_size = net->n_outputs;
    uint32_t lopt = neural_pred_lopt(xcsf);
    lopt &= ~LAYER_EVOLVE_NEURONS;
    lopt &= ~LAYER_EVOLVE_FUNCTIONS;
    l = neural_layer_connected_init(xcsf, code_size, xcsf->y_dim, xcsf->y_dim,
                                    LINEAR, lopt);
    neural_layer_insert(xcsf, net, l, net->n_layers);
    l = neural_layer_softmax_init(xcsf, xcsf->y_dim, 1);
    neural_layer_insert(xcsf, net, l, net->n_layers);
}
