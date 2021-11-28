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
 * @date 2020--2021.
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
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be initialised.
 */
void
act_neural_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    neural_create(&new->net, xcsf->act->largs);
    c->act = new;
}

/**
 * @brief Dummy function since crossover is not performed on neural actions.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose action is being crossed.
 * @param [in] c2 The second classifier whose action is being crossed.
 * @return False.
 */
bool
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
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The classifier whose action is tested to be more general.
 * @param [in] c2 The classifier whose action is tested to be more specific.
 * @return False.
 */
bool
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
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is being mutated.
 * @return Whether any alterations were made.
 */
bool
act_neural_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct ActNeural *act = c->act;
    return neural_mutate(&act->net);
}

/**
 * @brief Computes the current neural network action using the input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier calculating the action.
 * @param [in] x The input state.
 * @return The neural action.
 */
int
act_neural_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    struct ActNeural *act = c->act;
    neural_propagate(&act->net, x, xcsf->explore);
    const double *outputs = neural_outputs(&act->net);
    return max_index(outputs, xcsf->n_actions);
}

/**
 * @brief Copies a neural network action from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
void
act_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (void) xcsf;
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    const struct ActNeural *src_act = src->act;
    neural_copy(&new->net, &src_act->net);
    dest->act = new;
}

/**
 * @brief Prints a neural network action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be printed.
 */
void
act_neural_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", act_neural_json_export(xcsf, c));
}

/**
 * @brief Generates a neural network that covers the specified input:action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is being covered.
 * @param [in] x The input state to cover.
 * @param [in] action The action to cover.
 */
void
act_neural_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                 const int action)
{
    const struct ActNeural *act = c->act;
    do {
        neural_rand(&act->net);
    } while (action != act_neural_compute(xcsf, c, x));
}

/**
 * @brief Frees the memory used by a neural network action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be freed.
 */
void
act_neural_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct ActNeural *act = c->act;
    neural_free(&act->net);
    free(c->act);
}

/**
 * @brief Dummy function since neural network actions are not updated.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be updated.
 * @param [in] x The input state.
 * @param [in] y The payoff value.
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
 * @brief Writes a neural network action to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
act_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct ActNeural *act = c->act;
    size_t s = neural_save(&act->net, fp);
    return s;
}

/**
 * @brief Reads a neural network action from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
act_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    struct ActNeural *new = malloc(sizeof(struct ActNeural));
    size_t s = neural_load(&new->net, fp);
    c->act = new;
    return s;
}

/**
 * @brief Returns a json formatted string representation of a neural action.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose action is to be returned.
 * @return String encoded in json format.
 */
const char *
act_neural_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct ActNeural *act = c->act;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "neural");
    cJSON *network = cJSON_Parse(neural_json_export(&act->net, false));
    cJSON_AddItemToObject(json, "network", network);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
