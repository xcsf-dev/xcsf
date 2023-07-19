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
 * @date 2015--2022.
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
    return ACT_TYPE_INVALID;
}

/**
 * @brief Initialises default action parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
action_param_defaults(struct XCSF *xcsf)
{
    action_param_set_type(xcsf, ACT_TYPE_INTEGER);
    act_neural_param_defaults(xcsf);
}

/**
 * @brief Returns a json formatted string of the action parameters.
 * @param [in] xcsf XCSF data structure.
 * @return String encoded in json format.
 */
char *
action_param_json_export(const struct XCSF *xcsf)
{
    const struct ArgsAct *act = xcsf->act;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", action_type_as_string(act->type));
    char *json_str = NULL;
    if (xcsf->act->type == ACT_TYPE_NEURAL) {
        json_str = layer_args_json_export(xcsf->act->largs);
    }
    if (json_str != NULL) {
        cJSON *params = cJSON_Parse(json_str);
        if (params != NULL) {
            cJSON_AddItemToObject(json, "args", params);
        }
        free(json_str);
    }
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Sets the action parameters from a cJSON object.
 * @param [in,out] xcsf XCSF data structure.
 * @param [in] json cJSON object.
 * @return NULL if successful; or the name of parameter if not found.
 */
char *
action_param_json_import(struct XCSF *xcsf, cJSON *json)
{
    char *ret = NULL;
    switch (xcsf->act->type) {
        case ACT_TYPE_INTEGER:
            break;
        case ACT_TYPE_NEURAL:
            ret = act_neural_param_json_import(xcsf, json->child);
            break;
        default:
            printf("action_param_json_import(): unknown type.\n");
            exit(EXIT_FAILURE);
    }
    return ret;
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

int
action_param_set_type_string(struct XCSF *xcsf, const char *a)
{
    const int type = action_type_as_int(a);
    if (type != ACT_TYPE_INVALID) {
        xcsf->act->type = type;
    }
    return type;
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
