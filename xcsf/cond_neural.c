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
 * @file cond_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief Multi-layer perceptron neural network condition functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "condition.h"
#include "cond_neural.h"

static uint32_t cond_neural_lopt(const XCSF *xcsf);

void cond_neural_init(const XCSF *xcsf, CL *c)
{
    COND_NEURAL *new = malloc(sizeof(COND_NEURAL));
    neural_init(xcsf, &new->net);
    // hidden layers
    uint32_t lopt = cond_neural_lopt(xcsf);
    LAYER *l;
    int i = 0;
    int n_inputs = xcsf->x_dim;
    while(i < MAX_LAYERS && xcsf->COND_NUM_NEURONS[i] > 0) {
        int hinit = xcsf->COND_NUM_NEURONS[i];
        int hmax = xcsf->COND_MAX_NEURONS[i];
        if(hmax < hinit || !xcsf->COND_EVOLVE_NEURONS) {
            hmax = hinit;
        }
        int f = xcsf->COND_HIDDEN_ACTIVATION;
        l = neural_layer_connected_init(xcsf, n_inputs, hinit, hmax, f, lopt);
        neural_layer_insert(xcsf, &new->net, l, i);
        n_inputs = hinit;
        i++;
    }
    // output layer
    int f = xcsf->COND_OUTPUT_ACTIVATION;
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    l = neural_layer_connected_init(xcsf, n_inputs, 1, 1, f, lopt);
    neural_layer_insert(xcsf, &new->net, l, i);
    c->cond = new;
}

static uint32_t cond_neural_lopt(const XCSF *xcsf)
{
    uint32_t lopt = 0;
    if(xcsf->COND_EVOLVE_WEIGHTS) {
        lopt |= LAYER_EVOLVE_WEIGHTS;
    }
    if(xcsf->COND_EVOLVE_NEURONS) {
        lopt |= LAYER_EVOLVE_NEURONS;
    }
    if(xcsf->COND_EVOLVE_FUNCTIONS) {
        lopt |= LAYER_EVOLVE_FUNCTIONS;
    }
    if(xcsf->COND_EVOLVE_CONNECTIVITY) {
        lopt |= LAYER_EVOLVE_CONNECT;
    }
    return lopt;
}

void cond_neural_free(const XCSF *xcsf, const CL *c)
{
    COND_NEURAL *cond = c->cond;
    neural_free(xcsf, &cond->net);
    free(c->cond);
}

void cond_neural_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    COND_NEURAL *new = malloc(sizeof(COND_NEURAL));
    const COND_NEURAL *src_cond = src->cond;
    neural_copy(xcsf, &new->net, &src_cond->net);
    dest->cond = new;
}

void cond_neural_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_NEURAL *cond = c->cond;
    do {
        neural_rand(xcsf, &cond->net);
    } while(!cond_neural_match(xcsf, c, x));
}

void cond_neural_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)xcsf;
    (void)c;
    (void)x;
    (void)y;
}

_Bool cond_neural_match(const XCSF *xcsf, const CL *c, const double *x)
{
    const COND_NEURAL *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if(neural_output(xcsf, &cond->net, 0) > 0.5) {
        return true;
    }
    return false;
}

_Bool cond_neural_mutate(const XCSF *xcsf, const CL *c)
{
    const COND_NEURAL *cond = c->cond;
    return neural_mutate(xcsf, &cond->net);
}

_Bool cond_neural_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}

_Bool cond_neural_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}

void cond_neural_print(const XCSF *xcsf, const CL *c)
{
    const COND_NEURAL *cond = c->cond;
    neural_print(xcsf, &cond->net, false);
}

int cond_neural_size(const XCSF *xcsf, const CL *c)
{
    const COND_NEURAL *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t cond_neural_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    const COND_NEURAL *cond = c->cond;
    size_t s = neural_save(xcsf, &cond->net, fp);
    return s;
}

size_t cond_neural_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    COND_NEURAL *new = malloc(sizeof(COND_NEURAL));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->cond = new;
    return s;
}

int cond_neural_neurons(const XCSF *xcsf, const CL *c, int layer)
{
    (void)xcsf;
    const COND_NEURAL *cond = c->cond;
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

int cond_neural_connections(const XCSF *xcsf, const CL *c, int layer)
{
    (void)xcsf;
    const COND_NEURAL *cond = c->cond;
    const NET *net = &cond->net;
    int i = 0;
    for(const LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        if(i == layer) {
            return iter->layer->n_active;
        }
        i++;
    }
    return 0;
}

int cond_neural_layers(const XCSF *xcsf, const CL *c)
{
    (void)xcsf;
    const COND_NEURAL *cond = c->pred;
    const NET *net = &cond->net;
    return net->n_layers;
}
