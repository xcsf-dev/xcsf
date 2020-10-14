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

/**
 * @brief Creates and initialises a neural network condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be initialised.
 */
void
cond_neural_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondNeural *new = malloc(sizeof(struct CondNeural));
    neural_init(&new->net);
    const struct LayerArgs *arg = xcsf->cond->largs;
    while (arg != NULL) {
        struct Layer *l = layer_init(arg);
        neural_push(&new->net, l);
        arg = arg->next;
    }
    c->cond = new;
}

void
cond_neural_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct CondNeural *cond = c->cond;
    neural_free(&cond->net);
    free(c->cond);
}

void
cond_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (void) xcsf;
    struct CondNeural *new = malloc(sizeof(struct CondNeural));
    const struct CondNeural *src_cond = src->cond;
    neural_copy(&new->net, &src_cond->net);
    dest->cond = new;
}

void
cond_neural_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct CondNeural *cond = c->cond;
    do {
        neural_rand(&cond->net);
    } while (!cond_neural_match(xcsf, c, x));
}

void
cond_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

bool
cond_neural_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct CondNeural *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if (neural_output(&cond->net, 0) > 0.5) {
        return true;
    }
    return false;
}

bool
cond_neural_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    return neural_mutate(&cond->net);
}

bool
cond_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

bool
cond_neural_general(const struct XCSF *xcsf, const struct Cl *c1,
                    const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

void
cond_neural_print(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    neural_print(&cond->net, false);
}

double
cond_neural_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    return neural_size(&cond->net);
}

size_t
cond_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    size_t s = neural_save(&cond->net, fp);
    return s;
}

size_t
cond_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    struct CondNeural *new = malloc(sizeof(struct CondNeural));
    size_t s = neural_load(&new->net, fp);
    c->cond = new;
    return s;
}

int
cond_neural_neurons(const struct XCSF *xcsf, const struct Cl *c, int layer)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    const struct Net *net = &cond->net;
    const struct Llist *iter = net->tail;
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
cond_neural_connections(const struct XCSF *xcsf, const struct Cl *c, int layer)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    const struct Net *net = &cond->net;
    const struct Llist *iter = net->tail;
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
cond_neural_layers(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    const struct Net *net = &cond->net;
    return net->n_layers;
}
