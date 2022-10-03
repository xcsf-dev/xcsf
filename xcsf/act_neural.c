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
 * @date 2020--2022.
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
    return argmax(outputs, xcsf->n_actions);
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
    char *json_str = act_neural_json_export(xcsf, c);
    printf("%s\n", json_str);
    free(json_str);
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
char *
act_neural_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct ActNeural *act = c->act;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "neural");
    char *network_str = neural_json_export(&act->net, false);
    cJSON *network = cJSON_Parse(network_str);
    free(network_str);
    cJSON_AddItemToObject(json, "network", network);
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Creates a neural action from a cJSON object.
 * @param [in] xcsf The XCSF data structure.
 * @param [in,out] c The classifier to initialise.
 * @param [in] json cJSON object.
 */
void
act_neural_json_import(const struct XCSF *xcsf, struct Cl *c, const cJSON *json)
{
    const cJSON *item = cJSON_GetObjectItem(json, "network");
    if (item == NULL) {
        printf("Import error: missing network\n");
        exit(EXIT_FAILURE);
    }
    struct ActNeural *act = c->act;
    neural_json_import(&act->net, xcsf->act->largs, item);
}

/**
 * @brief Initialises default neural action parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
act_neural_param_defaults(struct XCSF *xcsf)
{
    // hidden layer
    struct ArgsLayer *la = malloc(sizeof(struct ArgsLayer));
    layer_args_init(la);
    la->type = CONNECTED;
    la->n_inputs = xcsf->x_dim;
    la->n_init = 1;
    la->n_max = 100;
    la->max_neuron_grow = 1;
    la->function = LOGISTIC;
    la->evolve_weights = true;
    la->evolve_neurons = true;
    la->evolve_connect = true;
    xcsf->act->largs = la;
    // softmax output layer
    la->next = layer_args_copy(la);
    la->next->function = LINEAR;
    la->next->n_inputs = la->n_init;
    la->next->n_init = xcsf->n_actions;
    la->next->n_max = xcsf->n_actions;
    la->next->evolve_neurons = false;
    la->next->next = layer_args_copy(la->next);
    la->next->next->n_inputs = la->next->n_init;
    la->next->next->type = SOFTMAX;
    la->next->next->scale = 1;
}

/**
 * @brief Sets the neural network parameters from a cJSON object.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 * @return NULL if successful; or the name of parameter if not found.
 */
char *
act_neural_param_json_import(struct XCSF *xcsf, cJSON *json)
{
    layer_args_free(&xcsf->act->largs);
    for (cJSON *iter = json; iter != NULL; iter = iter->next) {
        struct ArgsLayer *larg = malloc(sizeof(struct ArgsLayer));
        layer_args_init(larg);
        larg->n_inputs = xcsf->x_dim;
        char *ret = layer_args_json_import(larg, iter->child);
        if (ret != NULL) {
            return ret;
        }
        if (xcsf->act->largs == NULL) {
            xcsf->act->largs = larg;
        } else {
            struct ArgsLayer *layer_iter = xcsf->act->largs;
            while (layer_iter->next != NULL) {
                layer_iter = layer_iter->next;
            }
            layer_iter->next = larg;
        }
    }
    layer_args_validate(xcsf->act->largs);
    return NULL;
}
