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
 * @file action.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Interface for classifier actions.
 */

#include "action.h"
#include "act_integer.h"
#include "act_neural.h"
#include "utils.h"

/**
 * @brief Sets a classifier's action functions to the implementations.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to set.
 */
void
action_set(const struct XCSF *xcsf, struct Cl *c)
{
    switch (xcsf->act->type) {
        case ACT_TYPE_INTEGER:
            c->act_vptr = &act_integer_vtbl;
            break;
        case ACT_TYPE_NEURAL:
            c->act_vptr = &act_neural_vtbl;
            break;
        default:
            printf("Invalid action type specified: %d\n", xcsf->act->type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns a string representation of an action type from an integer.
 * @param [in] type Integer representation of an action type.
 * @return String representing the name of the action type.
 */
const char *
action_type_as_string(const int type)
{
    switch (type) {
        case ACT_TYPE_INTEGER:
            return ACT_STRING_INTEGER;
        case ACT_TYPE_NEURAL:
            return ACT_STRING_NEURAL;
        default:
            printf("action_type_as_string(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the integer representation of an action type given a name.
 * @param [in] type String representation of a condition type.
 * @return Integer representing the action type.
 */
int
action_type_as_int(const char *type)
{
    if (strncmp(type, ACT_STRING_INTEGER, 8) == 0) {
        return ACT_TYPE_INTEGER;
    }
    if (strncmp(type, ACT_STRING_NEURAL, 7) == 0) {
        return ACT_TYPE_NEURAL;
    }
    printf("action_type_as_int(): invalid type: %s\n", type);
    exit(EXIT_FAILURE);
}

/**
 * @brief Initialises default neural action parameters.
 * @param [in] xcsf The XCSF data structure.
 */
static void
action_param_defaults_neural(struct XCSF *xcsf)
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
 * @brief Initialises default action parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
action_param_defaults(struct XCSF *xcsf)
{
    action_param_set_type(xcsf, ACT_TYPE_NEURAL);
    action_param_defaults_neural(xcsf);
}

/**
 * @brief Returns a json formatted string of the action parameters.
 * @param [in] xcsf XCSF data structure.
 * @return String encoded in json format.
 */
const char *
action_param_json_export(const struct XCSF *xcsf)
{
    const struct ArgsAct *act = xcsf->act;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", action_type_as_string(act->type));
    cJSON *params = NULL;
    if (xcsf->act->type == ACT_TYPE_NEURAL) {
        params = cJSON_Parse(layer_args_json_export(xcsf->act->largs));
    }
    if (params != NULL) {
        cJSON_AddItemToObject(json, "args", params);
    }
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Saves action parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
action_param_save(const struct XCSF *xcsf, FILE *fp)
{
    const struct ArgsAct *act = xcsf->act;
    size_t s = 0;
    s += fwrite(&act->type, sizeof(int), 1, fp);
    s += layer_args_save(act->largs, fp);
    return s;
}

/**
 * @brief Loads action parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
action_param_load(struct XCSF *xcsf, FILE *fp)
{
    struct ArgsAct *act = xcsf->act;
    size_t s = 0;
    s += fread(&act->type, sizeof(int), 1, fp);
    s += layer_args_load(&act->largs, fp);
    return s;
}

/**
 * @brief Frees action parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
action_param_free(struct XCSF *xcsf)
{
    layer_args_free(&xcsf->act->largs);
}

/* parameter setters */

void
action_param_set_type_string(struct XCSF *xcsf, const char *a)
{
    xcsf->act->type = action_type_as_int(a);
}

void
action_param_set_type(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set ACT TYPE too small\n");
        xcsf->act->type = 0;
    } else {
        xcsf->act->type = a;
    }
}
