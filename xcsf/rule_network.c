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
 * @file rule_network.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Neural network rule (condition + prediction) functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_noise.h"
#include "neural_layer_softmax.h"
#include "cl.h"
#include "condition.h"
#include "prediction.h"
#include "rule_network.h"

/* CONDITION FUNCTIONS */

static uint32_t rule_network_lopt(const XCSF *xcsf);

void rule_network_cond_init(const XCSF *xcsf, CL *c)
{
    RULE_NETWORK *new = malloc(sizeof(RULE_NETWORK));
    neural_init(xcsf, &new->net);
    // hidden layers
    uint32_t lopt = rule_network_lopt(xcsf);
    LAYER *l;
    int i = 0;
    int n_inputs = xcsf->x_dim;
    while(i < MAX_LAYERS && xcsf->PRED_NUM_NEURONS[i] > 0) {
        int hinit = xcsf->PRED_NUM_NEURONS[i];
        int hmax = xcsf->PRED_MAX_NEURONS[i];
        if(hmax < hinit || !xcsf->PRED_EVOLVE_NEURONS) {
            hmax = hinit;
        }
        int f = xcsf->PRED_HIDDEN_ACTIVATION;
        l = neural_layer_connected_init(xcsf, n_inputs, hinit, hmax, f, lopt);
        neural_layer_insert(xcsf, &new->net, l, i);
        n_inputs = hinit;
        i++;
    }
    // output layer
    int f = xcsf->PRED_OUTPUT_ACTIVATION;
    int n_outputs = xcsf->y_dim + 1;
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    lopt &= ~LAYER_EVOLVE_FUNCTIONS; // never evolve the output neurons function
    if(f == SOFT_MAX) {
        // classification
        l = neural_layer_connected_init(xcsf, n_inputs, n_outputs, n_outputs, LINEAR, lopt);
        neural_layer_insert(xcsf, &new->net, l, i);
        l = neural_layer_softmax_init(xcsf, n_outputs, 1);
        neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
    } else {
        // regression
        l = neural_layer_connected_init(xcsf, n_inputs, n_outputs, n_outputs, f, lopt);
        neural_layer_insert(xcsf, &new->net, l, i);
    }
    c->cond = new;
}

static uint32_t rule_network_lopt(const XCSF *xcsf)
{
    uint32_t lopt = 0;
    if(xcsf->PRED_EVOLVE_ETA) {
        lopt |= LAYER_EVOLVE_ETA;
    }
    if(xcsf->PRED_SGD_WEIGHTS) {
        lopt |= LAYER_SGD_WEIGHTS;
    }
    if(xcsf->PRED_EVOLVE_WEIGHTS) {
        lopt |= LAYER_EVOLVE_WEIGHTS;
    }
    if(xcsf->PRED_EVOLVE_NEURONS) {
        lopt |= LAYER_EVOLVE_NEURONS;
    }
    if(xcsf->PRED_EVOLVE_FUNCTIONS) {
        lopt |= LAYER_EVOLVE_FUNCTIONS;
    }
    if(xcsf->PRED_EVOLVE_CONNECTIVITY) {
        lopt |= LAYER_EVOLVE_CONNECT;
    }
    return lopt;
}

void rule_network_cond_free(const XCSF *xcsf, const CL *c)
{
    RULE_NETWORK *cond = c->cond;
    neural_free(xcsf, &cond->net);
    free(c->cond);
}

void rule_network_cond_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    RULE_NETWORK *new = malloc(sizeof(RULE_NETWORK));
    const RULE_NETWORK *src_cond = src->cond;
    neural_copy(xcsf, &new->net, &src_cond->net);
    dest->cond = new;
}

void rule_network_cond_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    (void)xcsf;
    (void)c;
    (void)x;
}

void rule_network_cond_update(const XCSF *xcsf, const CL *c, const double *x,
                              const double *y)
{
    (void)xcsf;
    (void)c;
    (void)x;
    (void)y;
}

_Bool rule_network_cond_match(const XCSF *xcsf, const CL *c, const double *x)
{
    RULE_NETWORK *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if(neural_output(xcsf, &cond->net, xcsf->y_dim) > 0.5) {
        return true;
    }
    return false;
}

_Bool rule_network_cond_mutate(const XCSF *xcsf, const CL *c)
{
    const RULE_NETWORK *cond = c->cond;
    return neural_mutate(xcsf, &cond->net);
}

_Bool rule_network_cond_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}

_Bool rule_network_cond_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}

void rule_network_cond_print(const XCSF *xcsf, const CL *c)
{
    const RULE_NETWORK *cond = c->cond;
    neural_print(xcsf, &cond->net, false);
}

int rule_network_cond_size(const XCSF *xcsf, const CL *c)
{
    const RULE_NETWORK *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t rule_network_cond_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    const RULE_NETWORK *cond = c->cond;
    size_t s = neural_save(xcsf, &cond->net, fp);
    return s;
}

size_t rule_network_cond_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_NETWORK *new = malloc(sizeof(RULE_NETWORK));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->cond = new;
    return s;
}

/* PREDICTION FUNCTIONS */

void rule_network_pred_init(const XCSF *xcsf, CL *c)
{
    (void)xcsf;
    (void)c;
}

void rule_network_pred_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    (void)c;
}

void rule_network_pred_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    (void)xcsf;
    (void)dest;
    (void)src;
}

void rule_network_pred_update(const XCSF *xcsf, const CL *c, const double *x,
                              const double *y)
{
    if(xcsf->PRED_SGD_WEIGHTS) {
        RULE_NETWORK *cond = c->cond;
        neural_learn(xcsf, &cond->net, y, x);
    }
}

void rule_network_pred_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    (void)x;
    // assumes already forward propagated during matching
    const RULE_NETWORK *cond = c->cond;
    for(int i = 0; i < xcsf->y_dim; i++) {
        c->prediction[i] = neural_output(xcsf, &cond->net, i);
    }
}

void rule_network_pred_print(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    (void)c;
}

_Bool rule_network_pred_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}

_Bool rule_network_pred_mutate(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    (void)c;
    return false;
}

int rule_network_pred_size(const XCSF *xcsf, const CL *c)
{
    const RULE_NETWORK *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t rule_network_pred_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf;
    (void)c;
    (void)fp;
    return 0;
}

size_t rule_network_pred_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf;
    (void)c;
    (void)fp;
    return 0;
}

double rule_network_eta(const XCSF *xcsf, const CL *c, int layer)
{
    (void)xcsf;
    const RULE_NETWORK *cond = c->cond;
    int i = 0;
    for(const LLIST *iter = cond->net.tail; iter != NULL; iter = iter->prev) {
        if(i == layer) {
            return iter->layer->eta;
        }
        i++;
    }
    return 0;
}

int rule_network_neurons(const XCSF *xcsf, const CL *c, int layer)
{
    (void)xcsf;
    const RULE_NETWORK *cond = c->cond;
    const NET *net = &cond->net;
    int i = 0;
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        if(i == layer) {
            return iter->layer->n_outputs;
        }
        i++;
    }
    return 0;
}

int rule_network_layers(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const RULE_NETWORK *cond = c->cond;
    const NET *net = &cond->net;
    return net->n_layers;
}
