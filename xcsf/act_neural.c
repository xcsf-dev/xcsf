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
#include "utils.h"

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
        neural_push(xcsf, &new->net, l);
        n_inputs = hinit;
    }
    // output layer
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    lopt &= ~LAYER_EVOLVE_FUNCTIONS; // never evolve the output neurons function
    l = neural_layer_connected_init(xcsf, n_inputs, xcsf->n_actions,
                                    xcsf->n_actions, LINEAR, lopt);
    neural_push(xcsf, &new->net, l);
    l = neural_layer_softmax_init(xcsf, xcsf->n_actions, 1);
    neural_push(xcsf, &new->net, l);
    c->act = new;
}

/**
 * @brief Dummy function since crossover is not performed on neural actions.
 * @param xcsf The XCSF data structure.
 * @param c1 The first classifier whose action is being crossed.
 * @param c2 The second classifier whose action is being crossed.
 * @return False.
 */
_Bool
act_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Dummy function since neural actions do not generalise another.
 * @param xcsf The XCSF data structure.
 * @param c1 The classifier whose action is tested to be more general.
 * @param c2 The classifier whose action is tested to be more specific.
 * @return False.
 */
_Bool
act_neural_general(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Mutates a neural network action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is being mutated.
 * @return Whether any alterations were made.
 */
_Bool
act_neural_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct ActNeural *act = c->act;
    return neural_mutate(xcsf, &act->net);
}

/**
 * @brief Computes the current neural network action using the input.
 * @param xcsf The XCSF data structure.
 * @param c The classifier calculating the action.
 * @param x The input state.
 * @return The neural action.
 */
int
act_neural_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct ActNeural *act = c->act;
    neural_propagate(xcsf, &act->net, x);
    const double *outputs = neural_outputs(xcsf, &act->net);
    return max_index(outputs, xcsf->n_actions);
}

/**
 * @brief Copies a neural network action from one classifier to another.
 * @param xcsf The XCSF data structure.
 * @param dest The destination classifier.
 * @param src The source classifier.
 */
void
act_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    const struct ActNeural *src_act = src->act;
    neural_copy(xcsf, &new->net, &src_act->net);
    dest->act = new;
}

/**
 * @brief Prints a neural network action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be printed.
 */
void
act_neural_print(const struct XCSF *xcsf, const struct Cl *c)
{
    const struct ActNeural *act = c->act;
    neural_print(xcsf, &act->net, false);
}

/**
 * @brief Generates a neural network that covers the specified input:action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is being covered.
 * @param x The input state to cover.
 * @param action The action to cover.
 */
void
act_neural_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                 const int action)
{
    const struct ActNeural *act = c->act;
    do {
        neural_rand(xcsf, &act->net);
    } while (action != act_neural_compute(xcsf, c, x));
}

/**
 * @brief Frees the memory used by a neural network action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be freed.
 */
void
act_neural_free(const struct XCSF *xcsf, const struct Cl *c)
{
    struct ActNeural *act = c->act;
    neural_free(xcsf, &act->net);
    free(c->act);
}

/**
 * @brief Dummy function since neural network actions are not updated.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be updated.
 * @param x The input state.
 * @param y The payoff value.
 */
void
act_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                  const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

/**
 * @brief Writes a neural network action to a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be written.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
act_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    const struct ActNeural *act = c->act;
    size_t s = neural_save(xcsf, &act->net, fp);
    return s;
}

/**
 * @brief Reads a neural network action from a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be read.
 * @param fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
act_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->act = new;
    return s;
}
