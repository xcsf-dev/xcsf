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
 * @file act_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2020.
 * @brief Neural network action functions.
 */

#include "act_neural.h"
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
 * @brief Creates and initialises an action neural network.
 * @details Uses fully-connected layers.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be initialised.
 */
void
act_neural_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    neural_init(xcsf, &new->net);
    // hidden layers
    uint32_t lopt = neural_cond_lopt(xcsf);
    struct Layer *l = NULL;
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
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    lopt &= ~LAYER_EVOLVE_FUNCTIONS; // never evolve the output neurons function
    l = neural_layer_connected_init(xcsf, n_inputs, xcsf->n_actions,
                                    xcsf->n_actions, LINEAR, lopt);
    neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
    l = neural_layer_softmax_init(xcsf, xcsf->n_actions, 1);
    neural_layer_insert(xcsf, &new->net, l, new->net.n_layers);
    c->act = new;
}

_Bool
act_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
act_neural_general(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

_Bool
act_neural_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct ActNeural *act = c->act;
    return neural_mutate(xcsf, &act->net);
}

int
act_neural_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct ActNeural *act = c->act;
    neural_propagate(xcsf, &act->net, x);
    int action = 0;
    double max = neural_output(xcsf, &act->net, 0);
    for (int i = 1; i < xcsf->n_actions; ++i) {
        const double output = neural_output(xcsf, &act->net, i);
        if (output > max) {
            action = i;
            max = output;
        }
    }
    return action;
}

void
act_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    const struct ActNeural *src_act = src->act;
    neural_copy(xcsf, &new->net, &src_act->net);
    dest->act = new;
}

void
act_neural_print(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct ActNeural *act = c->act;
    neural_print(xcsf, &act->net, false);
}

void
act_neural_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                 const int action)
{
    const struct ActNeural *act = c->act;
    do {
        neural_rand(xcsf, &act->net);
    } while (action != act_neural_compute(xcsf, c, x));
}

void
act_neural_free(const struct XCSF *xcsf, const struct Cl *c)
{
    struct ActNeural *act = c->act;
    neural_free(xcsf, &act->net);
    free(c->act);
}

void
act_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                  const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

size_t
act_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    const struct ActNeural *act = c->act;
    size_t s = neural_save(xcsf, &act->net, fp);
    return s;
}

size_t
act_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->act = new;
    return s;
}
