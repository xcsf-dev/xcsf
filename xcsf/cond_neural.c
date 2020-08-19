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

#include "cond_neural.h"
#include "neural_activations.h"
#include "neural_layer_connected.h"
#include "neural_layer_convolutional.h"
#include "neural_layer_dropout.h"
#include "neural_layer_lstm.h"
#include "neural_layer_maxpool.h"
#include "neural_layer_noise.h"
#include "neural_layer_recurrent.h"
#include "neural_layer_softmax.h"

static uint32_t
cond_neural_lopt(const struct XCSF *xcsf)
{
    uint32_t lopt = 0;
    if (xcsf->COND_EVOLVE_WEIGHTS) {
        lopt |= LAYER_EVOLVE_WEIGHTS;
    }
    if (xcsf->COND_EVOLVE_NEURONS) {
        lopt |= LAYER_EVOLVE_NEURONS;
    }
    if (xcsf->COND_EVOLVE_FUNCTIONS) {
        lopt |= LAYER_EVOLVE_FUNCTIONS;
    }
    if (xcsf->COND_EVOLVE_CONNECTIVITY) {
        lopt |= LAYER_EVOLVE_CONNECT;
    }
    return lopt;
}

/**
 * @brief Creates and initialises a neural network condition.
 * @details Uses fully-connected layers.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose condition is to be initialised.
 */
void
cond_neural_init(const struct XCSF *xcsf, struct CL *c)
{
    struct COND_NEURAL *new = malloc(sizeof(struct COND_NEURAL));
    neural_init(xcsf, &new->net);
    // hidden layers
    uint32_t lopt = cond_neural_lopt(xcsf);
    struct LAYER *l = NULL;
    int n_inputs = xcsf->x_dim;
    for (int i = 0; i < MAX_LAYERS && xcsf->COND_NUM_NEURONS[i] > 0; ++i) {
        const int hinit = xcsf->COND_NUM_NEURONS[i];
        int hmax = xcsf->COND_MAX_NEURONS[i];
        if (hmax < hinit || !xcsf->COND_EVOLVE_NEURONS) {
            hmax = hinit;
        }
        const int f = xcsf->COND_HIDDEN_ACTIVATION;
        l = neural_layer_connected_init(xcsf, n_inputs, hinit, hmax, f, lopt);
        neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
        n_inputs = hinit;
    }
    // output layer
    const int f = xcsf->COND_OUTPUT_ACTIVATION;
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    l = neural_layer_connected_init(xcsf, n_inputs, 1, 1, f, lopt);
    neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
    c->cond = new;
}

void
cond_neural_free(const struct XCSF *xcsf, const struct CL *c)
{
    struct COND_NEURAL *cond = c->cond;
    neural_free(xcsf, &cond->net);
    free(c->cond);
}

void
cond_neural_copy(const struct XCSF *xcsf, struct CL *dest, const struct CL *src)
{
    struct COND_NEURAL *new = malloc(sizeof(struct COND_NEURAL));
    const struct COND_NEURAL *src_cond = src->cond;
    neural_copy(xcsf, &new->net, &src_cond->net);
    dest->cond = new;
}

void
cond_neural_cover(const struct XCSF *xcsf, const struct CL *c, const double *x)
{
    const struct COND_NEURAL *cond = c->cond;
    do {
        neural_rand(xcsf, &cond->net);
    } while (!cond_neural_match(xcsf, c, x));
}

void
cond_neural_update(const struct XCSF *xcsf, const struct CL *c, const double *x,
                   const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

_Bool
cond_neural_match(const struct XCSF *xcsf, const struct CL *c, const double *x)
{
    const struct COND_NEURAL *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if (neural_output(xcsf, &cond->net, 0) > 0.5) {
        return true;
    }
    return false;
}

_Bool
cond_neural_mutate(const struct XCSF *xcsf, const struct CL *c)
{
    const struct COND_NEURAL *cond = c->cond;
    return neural_mutate(xcsf, &cond->net);
}

_Bool
cond_neural_crossover(const struct XCSF *xcsf, const struct CL *c1,
                      const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
cond_neural_general(const struct XCSF *xcsf, const struct CL *c1,
                    const struct CL *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
cond_neural_print(const struct XCSF *xcsf, const struct CL *c)
{
    const struct COND_NEURAL *cond = c->cond;
    neural_print(xcsf, &cond->net, false);
}

double
cond_neural_size(const struct XCSF *xcsf, const struct CL *c)
{
    const struct COND_NEURAL *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t
cond_neural_save(const struct XCSF *xcsf, const struct CL *c, FILE *fp)
{
    const struct COND_NEURAL *cond = c->cond;
    size_t s = neural_save(xcsf, &cond->net, fp);
    return s;
}

size_t
cond_neural_load(const struct XCSF *xcsf, struct CL *c, FILE *fp)
{
    struct COND_NEURAL *new = malloc(sizeof(struct COND_NEURAL));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->cond = new;
    return s;
}

int
cond_neural_neurons(const struct XCSF *xcsf, const struct CL *c, int layer)
{
    (void) xcsf;
    const struct COND_NEURAL *cond = c->cond;
    const struct NET *net = &cond->net;
    const struct LLIST *iter = net->tail;
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
cond_neural_connections(const struct XCSF *xcsf, const struct CL *c, int layer)
{
    (void) xcsf;
    const struct COND_NEURAL *cond = c->cond;
    const struct NET *net = &cond->net;
    const struct LLIST *iter = net->tail;
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
cond_neural_layers(const struct XCSF *xcsf, const struct CL *c)
{
    (void) xcsf;
    const struct COND_NEURAL *cond = c->cond;
    const struct NET *net = &cond->net;
    return net->n_layers;
}
