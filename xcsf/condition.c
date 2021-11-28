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
 * @file condition.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Interface for classifier conditions.
 */

#include "cond_dgp.h"
#include "cond_dummy.h"
#include "cond_ellipsoid.h"
#include "cond_gp.h"
#include "cond_neural.h"
#include "cond_rectangle.h"
#include "cond_ternary.h"
#include "rule_dgp.h"
#include "rule_neural.h"
#include "utils.h"

/**
 * @brief Sets a classifier's condition functions to the implementations.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to set.
 */
void
condition_set(const struct XCSF *xcsf, struct Cl *c)
{
    switch (xcsf->cond->type) {
        case COND_TYPE_DUMMY:
            c->cond_vptr = &cond_dummy_vtbl;
            break;
        case COND_TYPE_HYPERRECTANGLE:
            c->cond_vptr = &cond_rectangle_vtbl;
            break;
        case COND_TYPE_HYPERELLIPSOID:
            c->cond_vptr = &cond_ellipsoid_vtbl;
            break;
        case COND_TYPE_NEURAL:
            c->cond_vptr = &cond_neural_vtbl;
            break;
        case COND_TYPE_GP:
            c->cond_vptr = &cond_gp_vtbl;
            break;
        case COND_TYPE_DGP:
            c->cond_vptr = &cond_dgp_vtbl;
            break;
        case COND_TYPE_TERNARY:
            c->cond_vptr = &cond_ternary_vtbl;
            break;
        case RULE_TYPE_DGP:
            c->cond_vptr = &rule_dgp_cond_vtbl;
            c->act_vptr = &rule_dgp_act_vtbl;
            break;
        case RULE_TYPE_NEURAL:
            c->cond_vptr = &rule_neural_cond_vtbl;
            c->act_vptr = &rule_neural_act_vtbl;
            break;
        default:
            printf("Invalid condition type specified: %d\n", xcsf->cond->type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns a string representation of a condition type from an integer.
 * @param [in] type Integer representation of a condition type.
 * @return String representing the name of the condition type.
 */
const char *
condition_type_as_string(const int type)
{
    switch (type) {
        case COND_TYPE_DUMMY:
            return COND_STRING_DUMMY;
        case COND_TYPE_HYPERRECTANGLE:
            return COND_STRING_HYPERRECTANGLE;
        case COND_TYPE_HYPERELLIPSOID:
            return COND_STRING_HYPERELLIPSOID;
        case COND_TYPE_NEURAL:
            return COND_STRING_NEURAL;
        case COND_TYPE_GP:
            return COND_STRING_GP;
        case COND_TYPE_DGP:
            return COND_STRING_DGP;
        case COND_TYPE_TERNARY:
            return COND_STRING_TERNARY;
        case RULE_TYPE_DGP:
            return COND_STRING_RULE_DGP;
        case RULE_TYPE_NEURAL:
            return COND_STRING_RULE_NEURAL;
        case RULE_TYPE_NETWORK:
            return COND_STRING_RULE_NETWORK;
        default:
            printf("condition_type_as_string(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the integer representation of a condition type given a name.
 * @param [in] type String representation of a condition type.
 * @return Integer representing the condition type.
 */
int
condition_type_as_int(const char *type)
{
    if (strncmp(type, COND_STRING_DUMMY, 6) == 0) {
        return COND_TYPE_DUMMY;
    }
    if (strncmp(type, COND_STRING_HYPERRECTANGLE, 15) == 0) {
        return COND_TYPE_HYPERRECTANGLE;
    }
    if (strncmp(type, COND_STRING_HYPERELLIPSOID, 15) == 0) {
        return COND_TYPE_HYPERELLIPSOID;
    }
    if (strncmp(type, COND_STRING_NEURAL, 7) == 0) {
        return COND_TYPE_NEURAL;
    }
    if (strncmp(type, COND_STRING_GP, 8) == 0) {
        return COND_TYPE_GP;
    }
    if (strncmp(type, COND_STRING_DGP, 4) == 0) {
        return COND_TYPE_DGP;
    }
    if (strncmp(type, COND_STRING_TERNARY, 8) == 0) {
        return COND_TYPE_TERNARY;
    }
    if (strncmp(type, COND_STRING_RULE_DGP, 9) == 0) {
        return RULE_TYPE_DGP;
    }
    if (strncmp(type, COND_STRING_RULE_NEURAL, 12) == 0) {
        return RULE_TYPE_NEURAL;
    }
    if (strncmp(type, COND_STRING_RULE_NETWORK, 13) == 0) {
        return RULE_TYPE_NETWORK;
    }
    printf("condition_type_as_int(): invalid type: %s\n", type);
    exit(EXIT_FAILURE);
}

/**
 * @brief Initialises default tree GP condition parameters.
 * @param [in] xcsf The XCSF data structure.
 */
static void
cond_param_defaults_gp(struct XCSF *xcsf)
{
    struct ArgsGPTree *args = malloc(sizeof(struct ArgsGPTree));
    tree_args_init(args);
    tree_param_set_max(args, 1);
    tree_param_set_min(args, 0);
    tree_param_set_init_depth(args, 5);
    tree_param_set_max_len(args, 10000);
    tree_param_set_n_constants(args, 100);
    tree_param_set_n_inputs(args, xcsf->x_dim);
    tree_args_init_constants(args);
    xcsf->cond->targs = args;
}

/**
 * @brief Initialises default DGP condition parameters.
 * @param [in] xcsf The XCSF data structure.
 */
static void
cond_param_defaults_dgp(struct XCSF *xcsf)
{
    struct ArgsDGP *args = malloc(sizeof(struct ArgsDGP));
    graph_args_init(args);
    graph_param_set_max_k(args, 2);
    graph_param_set_max_t(args, 10);
    graph_param_set_n(args, 10);
    graph_param_set_n_inputs(args, xcsf->x_dim);
    graph_param_set_evolve_cycles(args, true);
    xcsf->cond->dargs = args;
}

/**
 * @brief Initialises default neural condition parameters.
 * @param [in] xcsf The XCSF data structure.
 */
static void
cond_param_defaults_neural(struct XCSF *xcsf)
{
    // hidden layer
    struct ArgsLayer *args = malloc(sizeof(struct ArgsLayer));
    layer_args_init(args);
    args->type = CONNECTED;
    args->n_inputs = xcsf->x_dim;
    args->n_init = 10;
    args->n_max = 100;
    args->max_neuron_grow = 1;
    args->function = LOGISTIC;
    args->evolve_weights = true;
    args->evolve_neurons = true;
    args->evolve_connect = true;
    xcsf->cond->largs = args;
    // output layer
    args->next = layer_args_copy(args);
    args->next->function = LINEAR;
    args->next->n_inputs = args->n_init;
    args->next->n_max = 1;
    args->next->evolve_neurons = false;
}

/**
 * @brief Initialises default condition parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
cond_param_defaults(struct XCSF *xcsf)
{
    cond_param_set_type(xcsf, COND_TYPE_HYPERRECTANGLE);
    cond_param_set_eta(xcsf, 0);
    cond_param_set_min(xcsf, 0);
    cond_param_set_max(xcsf, 1);
    cond_param_set_spread_min(xcsf, 0.1);
    cond_param_set_p_dontcare(xcsf, 0.5);
    cond_param_set_bits(xcsf, 1);
    cond_param_defaults_neural(xcsf);
    cond_param_defaults_dgp(xcsf);
    cond_param_defaults_gp(xcsf);
}

/**
 * @brief Returns a json formatted string of the ternary parameters.
 * @param [in] xcsf The XCSF data structure.
 * @return String encoded in json format.
 */
static const char *
cond_param_json_export_ternary(const struct XCSF *xcsf)
{
    const struct ArgsCond *cond = xcsf->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "p_dontcare", cond->p_dontcare);
    cJSON_AddNumberToObject(json, "bits", cond->bits);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Returns a json formatted string of the center-spread parameters.
 * @param [in] xcsf The XCSF data structure.
 * @return String encoded in json format.
 */
static const char *
cond_param_json_export_csr(const struct XCSF *xcsf)
{
    const struct ArgsCond *cond = xcsf->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "eta", cond->eta);
    cJSON_AddNumberToObject(json, "min", cond->min);
    cJSON_AddNumberToObject(json, "max", cond->max);
    cJSON_AddNumberToObject(json, "spread_min", cond->spread_min);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Returns a json formatted string of the condition parameters.
 * @param [in] xcsf XCSF data structure.
 * @return String encoded in json format.
 */
const char *
cond_param_json_export(const struct XCSF *xcsf)
{
    const struct ArgsCond *cond = xcsf->cond;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", condition_type_as_string(cond->type));
    cJSON *params = NULL;
    switch (cond->type) {
        case COND_TYPE_TERNARY:
            params = cJSON_Parse(cond_param_json_export_ternary(xcsf));
            break;
        case COND_TYPE_HYPERELLIPSOID:
        case COND_TYPE_HYPERRECTANGLE:
            params = cJSON_Parse(cond_param_json_export_csr(xcsf));
            break;
        case COND_TYPE_GP:
            params = cJSON_Parse(tree_args_json_export(xcsf->cond->targs));
            break;
        case COND_TYPE_DGP:
        case RULE_TYPE_DGP:
            params = cJSON_Parse(graph_args_json_export(xcsf->cond->dargs));
            break;
        case COND_TYPE_NEURAL:
        case RULE_TYPE_NEURAL:
        case RULE_TYPE_NETWORK:
            params = cJSON_Parse(layer_args_json_export(xcsf->cond->largs));
            break;
        default:
            break;
    }
    if (params != NULL) {
        cJSON_AddItemToObject(json, "args", params);
    }
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Saves condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
cond_param_save(const struct XCSF *xcsf, FILE *fp)
{
    const struct ArgsCond *cond = xcsf->cond;
    size_t s = 0;
    s += fwrite(&cond->type, sizeof(int), 1, fp);
    s += fwrite(&cond->eta, sizeof(double), 1, fp);
    s += fwrite(&cond->min, sizeof(double), 1, fp);
    s += fwrite(&cond->max, sizeof(double), 1, fp);
    s += fwrite(&cond->spread_min, sizeof(double), 1, fp);
    s += fwrite(&cond->p_dontcare, sizeof(double), 1, fp);
    s += fwrite(&cond->bits, sizeof(int), 1, fp);
    s += graph_args_save(cond->dargs, fp);
    s += tree_args_save(cond->targs, fp);
    s += layer_args_save(cond->largs, fp);
    return s;
}

/**
 * @brief Loads condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
cond_param_load(struct XCSF *xcsf, FILE *fp)
{
    struct ArgsCond *cond = xcsf->cond;
    size_t s = 0;
    s += fread(&cond->type, sizeof(int), 1, fp);
    s += fread(&cond->eta, sizeof(double), 1, fp);
    s += fread(&cond->min, sizeof(double), 1, fp);
    s += fread(&cond->max, sizeof(double), 1, fp);
    s += fread(&cond->spread_min, sizeof(double), 1, fp);
    s += fread(&cond->p_dontcare, sizeof(double), 1, fp);
    s += fread(&cond->bits, sizeof(int), 1, fp);
    s += graph_args_load(cond->dargs, fp);
    s += tree_args_load(cond->targs, fp);
    s += layer_args_load(&cond->largs, fp);
    return s;
}

/**
 * @brief Frees condition parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
cond_param_free(struct XCSF *xcsf)
{
    tree_args_free(xcsf->cond->targs);
    free(xcsf->cond->targs);
    free(xcsf->cond->dargs);
    xcsf->cond->targs = NULL;
    xcsf->cond->dargs = NULL;
    layer_args_free(&xcsf->cond->largs);
}

/* parameter setters */

void
cond_param_set_eta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set COND ETA too small\n");
        xcsf->cond->eta = 0;
    } else if (a > 1) {
        printf("Warning: tried to set COND ETA too large\n");
        xcsf->cond->eta = 1;
    } else {
        xcsf->cond->eta = a;
    }
}

void
cond_param_set_min(struct XCSF *xcsf, const double a)
{
    xcsf->cond->min = a;
}

void
cond_param_set_max(struct XCSF *xcsf, const double a)
{
    xcsf->cond->max = a;
}

void
cond_param_set_p_dontcare(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set COND P_DONTCARE too small\n");
        xcsf->cond->p_dontcare = 0;
    } else if (a > 1) {
        printf("Warning: tried to set COND P_DONTCARE too large\n");
        xcsf->cond->p_dontcare = 1;
    } else {
        xcsf->cond->p_dontcare = a;
    }
}

void
cond_param_set_spread_min(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set COND SPREAD_MIN too small\n");
        xcsf->cond->spread_min = 0;
    } else {
        xcsf->cond->spread_min = a;
    }
}

void
cond_param_set_bits(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set COND BITS too small\n");
        xcsf->cond->bits = 1;
    } else {
        xcsf->cond->bits = a;
    }
}

void
cond_param_set_type_string(struct XCSF *xcsf, const char *a)
{
    xcsf->cond->type = condition_type_as_int(a);
}

void
cond_param_set_type(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set COND TYPE too small\n");
        xcsf->cond->type = 0;
    } else {
        xcsf->cond->type = a;
    }
}
