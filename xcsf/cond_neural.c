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
 * @date 2016--2021.
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
#include "utils.h"

/**
 * @brief Creates and initialises a neural network condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be initialised.
 */
void
cond_neural_init(const struct XCSF *xcsf, struct Cl *c)
{
    struct CondNeural *new = malloc(sizeof(struct CondNeural));
    neural_create(&new->net, xcsf->cond->largs);
    c->cond = new;
}

/**
 * @brief Frees the memory used by a neural network condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be freed.
 */
void
cond_neural_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    struct CondNeural *cond = c->cond;
    neural_free(&cond->net);
    free(c->cond);
}

/**
 * @brief Copies a neural network condition from one classifier to another.
 * @param [in] xcsf XCSF data structure.
 * @param [in] dest Destination classifier.
 * @param [in] src Source classifier.
 */
void
cond_neural_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (void) xcsf;
    struct CondNeural *new = malloc(sizeof(struct CondNeural));
    const struct CondNeural *src_cond = src->cond;
    neural_copy(&new->net, &src_cond->net);
    dest->cond = new;
}

/**
 * @brief Generates a neural network that matches the current input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
void
cond_neural_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    const struct CondNeural *cond = c->cond;
    do {
        neural_rand(&cond->net);
    } while (!cond_neural_match(xcsf, c, x));
}

/**
 * @brief Dummy update function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_neural_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                   const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

/**
 * @brief Generates a neural network that matches the current input.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
bool
cond_neural_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    struct CondNeural *cond = c->cond;
    neural_propagate(&cond->net, x, xcsf->explore);
    if (neural_output(&cond->net, 0) > 0.5) {
        return true;
    }
    return false;
}

/**
 * @brief Mutates a neural network condition with the self-adaptive rates.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
bool
cond_neural_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    return neural_mutate(&cond->net);
}

/**
 * @brief Dummy crossover function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 First classifier whose condition is being crossed.
 * @param [in] c2 Second classifier whose condition is being crossed.
 * @return False
 */
bool
cond_neural_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                      const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Dummy general function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 Classifier whose condition is tested to be more general.
 * @param [in] c2 Classifier whose condition is tested to be more specific.
 * @return False.
 */
bool
cond_neural_general(const struct XCSF *xcsf, const struct Cl *c1,
                    const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Prints a neural network condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be printed.
 */
void
cond_neural_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", cond_neural_json_export(xcsf, c));
}

/**
 * @brief Returns the size of a neural network condition.
 * @see neural_size()
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition size to return.
 * @return The network size.
 */
double
cond_neural_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    return neural_size(&cond->net);
}

/**
 * @brief Writes a neural network condition to a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
cond_neural_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    size_t s = neural_save(&cond->net, fp);
    return s;
}

/**
 * @brief Reads a neural network condition from a file.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
cond_neural_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    struct CondNeural *new = malloc(sizeof(struct CondNeural));
    size_t s = neural_load(&new->net, fp);
    c->cond = new;
    return s;
}

/**
 * @brief Returns the number of neurons in a neural condition layer.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier maintaining a neural network prediction.
 * @param [in] layer Position of a layer in the network.
 * @return The current number of neurons in a layer.
 */
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
 * @brief Returns the number of active connections in a neural condition layer.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier maintaining a neural network condition.
 * @param [in] layer Position of a layer in the network.
 * @return The current number of active connections in a layer.
 */
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

/**
 * @brief Returns the number of layers within a neural network condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier maintaining a neural network condition.
 * @return The number of layers.
 */
int
cond_neural_layers(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    const struct Net *net = &cond->net;
    return net->n_layers;
}

/**
 * @brief Returns a json formatted string representation of a neural condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
const char *
cond_neural_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    const struct CondNeural *cond = c->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "neural");
    cJSON *network = cJSON_Parse(neural_json_export(&cond->net, false));
    cJSON_AddItemToObject(json, "network", network);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
