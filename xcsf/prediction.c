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
 * @file prediction.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Interface for classifier predictions.
 */

#include "pred_constant.h"
#include "pred_neural.h"
#include "pred_nlms.h"
#include "pred_rls.h"
#include "utils.h"

/**
 * @brief Sets a classifier's prediction functions to the implementations.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to set.
 */
void
prediction_set(const struct XCSF *xcsf, struct Cl *c)
{
    switch (xcsf->pred->type) {
        case PRED_TYPE_CONSTANT:
            c->pred_vptr = &pred_constant_vtbl;
            break;
        case PRED_TYPE_NLMS_LINEAR:
        case PRED_TYPE_NLMS_QUADRATIC:
            c->pred_vptr = &pred_nlms_vtbl;
            break;
        case PRED_TYPE_RLS_LINEAR:
        case PRED_TYPE_RLS_QUADRATIC:
            c->pred_vptr = &pred_rls_vtbl;
            break;
        case PRED_TYPE_NEURAL:
            c->pred_vptr = &pred_neural_vtbl;
            break;
        default:
            printf("prediction_set(): invalid type: %d\n", xcsf->pred->type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns a string representation of a prediction type from the integer.
 * @param [in] type Integer representation of a prediction type.
 * @return String representing the name of the prediction type.
 */
const char *
prediction_type_as_string(const int type)
{
    switch (type) {
        case PRED_TYPE_CONSTANT:
            return PRED_STRING_CONSTANT;
        case PRED_TYPE_NLMS_LINEAR:
            return PRED_STRING_NLMS_LINEAR;
        case PRED_TYPE_NLMS_QUADRATIC:
            return PRED_STRING_NLMS_QUADRATIC;
        case PRED_TYPE_RLS_LINEAR:
            return PRED_STRING_RLS_LINEAR;
        case PRED_TYPE_RLS_QUADRATIC:
            return PRED_STRING_RLS_QUADRATIC;
        case PRED_TYPE_NEURAL:
            return PRED_STRING_NEURAL;
        default:
            printf("prediction_type_as_string(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the integer representation of a prediction type given a name.
 * @param [in] type String representation of a prediction type.
 * @return Integer representing the prediction type.
 */
int
prediction_type_as_int(const char *type)
{
    if (strncmp(type, PRED_STRING_CONSTANT, 9) == 0) {
        return PRED_TYPE_CONSTANT;
    }
    if (strncmp(type, PRED_STRING_NLMS_LINEAR, 12) == 0) {
        return PRED_TYPE_NLMS_LINEAR;
    }
    if (strncmp(type, PRED_STRING_NLMS_QUADRATIC, 15) == 0) {
        return PRED_TYPE_NLMS_QUADRATIC;
    }
    if (strncmp(type, PRED_STRING_RLS_LINEAR, 11) == 0) {
        return PRED_TYPE_RLS_LINEAR;
    }
    if (strncmp(type, PRED_STRING_RLS_QUADRATIC, 14) == 0) {
        return PRED_TYPE_RLS_QUADRATIC;
    }
    if (strncmp(type, PRED_STRING_NEURAL, 7) == 0) {
        return PRED_TYPE_NEURAL;
    }
    printf("prediction_type_as_int(): invalid type: %s\n", type);
    exit(EXIT_FAILURE);
}

/**
 * @brief Initialises default neural prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 */
static void
pred_param_defaults_neural(struct XCSF *xcsf)
{
    // hidden layer
    struct ArgsLayer *la = malloc(sizeof(struct ArgsLayer));
    layer_args_init(la);
    la->type = CONNECTED;
    la->n_inputs = xcsf->x_dim;
    la->n_init = 10;
    la->n_max = 100;
    la->max_neuron_grow = 1;
    la->function = LOGISTIC;
    la->evolve_weights = true;
    la->evolve_neurons = true;
    la->evolve_connect = true;
    la->evolve_eta = true;
    la->sgd_weights = true;
    la->eta = 0.01;
    la->momentum = 0.9;
    xcsf->pred->largs = la;
    // output layer
    la->next = layer_args_copy(la);
    la->next->n_inputs = la->n_init;
    la->next->n_init = xcsf->y_dim;
    la->next->n_max = xcsf->y_dim;
    la->next->evolve_neurons = false;
}

/**
 * @brief Initialises default prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
pred_param_defaults(struct XCSF *xcsf)
{
    pred_param_set_type(xcsf, PRED_TYPE_NLMS_LINEAR);
    pred_param_set_eta(xcsf, 0.1);
    pred_param_set_eta_min(xcsf, 0.00001);
    pred_param_set_lambda(xcsf, 1);
    pred_param_set_scale_factor(xcsf, 1000);
    pred_param_set_x0(xcsf, 1);
    pred_param_set_evolve_eta(xcsf, true);
    pred_param_defaults_neural(xcsf);
}

/**
 * @brief Returns a json formatted string of the NLMS parameters.
 * @param [in] xcsf The XCSF data structure.
 * @return String encoded in json format.
 */
static const char *
pred_param_json_export_nlms(const struct XCSF *xcsf)
{
    const struct ArgsPred *pred = xcsf->pred;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "x0", pred->x0);
    cJSON_AddNumberToObject(json, "eta", pred->eta);
    cJSON_AddBoolToObject(json, "evolve_eta", pred->evolve_eta);
    if (pred->evolve_eta) {
        cJSON_AddNumberToObject(json, "eta_min", pred->eta_min);
    }
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Returns a json formatted string of the RLS parameters.
 * @param [in] xcsf The XCSF data structure.
 * @return String encoded in json format.
 */
static const char *
pred_param_json_export_rls(const struct XCSF *xcsf)
{
    const struct ArgsPred *pred = xcsf->pred;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "x0", pred->x0);
    cJSON_AddNumberToObject(json, "lambda", pred->lambda);
    cJSON_AddNumberToObject(json, "scale_factor", pred->scale_factor);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Returns a json formatted string of the prediction parameters.
 * @param [in] xcsf XCSF data structure.
 * @return String encoded in json format.
 */
const char *
pred_param_json_export(const struct XCSF *xcsf)
{
    const struct ArgsPred *pred = xcsf->pred;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type",
                            prediction_type_as_string(pred->type));
    cJSON *params = NULL;
    switch (pred->type) {
        case PRED_TYPE_NLMS_LINEAR:
        case PRED_TYPE_NLMS_QUADRATIC:
            params = cJSON_Parse(pred_param_json_export_nlms(xcsf));
            break;
        case PRED_TYPE_RLS_LINEAR:
        case PRED_TYPE_RLS_QUADRATIC:
            params = cJSON_Parse(pred_param_json_export_rls(xcsf));
            break;
        case PRED_TYPE_NEURAL:
            params = cJSON_Parse(layer_args_json_export(xcsf->pred->largs));
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
 * @brief Saves prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
pred_param_save(const struct XCSF *xcsf, FILE *fp)
{
    const struct ArgsPred *pred = xcsf->pred;
    size_t s = 0;
    s += fwrite(&pred->type, sizeof(int), 1, fp);
    s += fwrite(&pred->eta, sizeof(double), 1, fp);
    s += fwrite(&pred->eta_min, sizeof(double), 1, fp);
    s += fwrite(&pred->lambda, sizeof(double), 1, fp);
    s += fwrite(&pred->scale_factor, sizeof(double), 1, fp);
    s += fwrite(&pred->x0, sizeof(double), 1, fp);
    s += fwrite(&pred->evolve_eta, sizeof(bool), 1, fp);
    s += layer_args_save(pred->largs, fp);
    return s;
}

/**
 * @brief Loads prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
pred_param_load(struct XCSF *xcsf, FILE *fp)
{
    struct ArgsPred *pred = xcsf->pred;
    size_t s = 0;
    s += fread(&pred->type, sizeof(int), 1, fp);
    s += fread(&pred->eta, sizeof(double), 1, fp);
    s += fread(&pred->eta_min, sizeof(double), 1, fp);
    s += fread(&pred->lambda, sizeof(double), 1, fp);
    s += fread(&pred->scale_factor, sizeof(double), 1, fp);
    s += fread(&pred->x0, sizeof(double), 1, fp);
    s += fread(&pred->evolve_eta, sizeof(bool), 1, fp);
    s += layer_args_load(&pred->largs, fp);
    return s;
}

/**
 * @brief Frees prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
pred_param_free(struct XCSF *xcsf)
{
    layer_args_free(&xcsf->pred->largs);
}

/**
 * @brief Prepares the input state for least squares computation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] x The input state.
 * @param [in] X0 Bias term.
 * @param [out] tmp_input The transformed input.
 */
void
pred_transform_input(const struct XCSF *xcsf, const double *x, const double X0,
                     double *tmp_input)
{
    // bias term
    tmp_input[0] = X0;
    int idx = 1;
    // linear terms
    for (int i = 0; i < xcsf->x_dim; ++i) {
        tmp_input[idx] = x[i];
        ++idx;
    }
    // quadratic terms
    if (xcsf->pred->type == PRED_TYPE_NLMS_QUADRATIC ||
        xcsf->pred->type == PRED_TYPE_RLS_QUADRATIC) {
        for (int i = 0; i < xcsf->x_dim; ++i) {
            for (int j = i; j < xcsf->x_dim; ++j) {
                tmp_input[idx] = x[i] * x[j];
                ++idx;
            }
        }
    }
}

/* parameter setters */

void
pred_param_set_eta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED ETA too small\n");
        xcsf->pred->eta = 0;
    } else if (a > 1) {
        printf("Warning: tried to set PRED ETA too large\n");
        xcsf->pred->eta = 1;
    } else {
        xcsf->pred->eta = a;
    }
}

void
pred_param_set_eta_min(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED ETA_MIN too small\n");
        xcsf->pred->eta_min = 0;
    } else if (a > 1) {
        printf("Warning: tried to set PRED ETA_MIN too large\n");
        xcsf->pred->eta_min = 1;
    } else {
        xcsf->pred->eta_min = a;
    }
}

void
pred_param_set_lambda(struct XCSF *xcsf, const double a)
{
    xcsf->pred->lambda = a;
}

void
pred_param_set_scale_factor(struct XCSF *xcsf, const double a)
{
    xcsf->pred->scale_factor = a;
}

void
pred_param_set_x0(struct XCSF *xcsf, const double a)
{
    xcsf->pred->x0 = a;
}

void
pred_param_set_evolve_eta(struct XCSF *xcsf, const bool a)
{
    xcsf->pred->evolve_eta = a;
}

void
pred_param_set_type(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set PRED TYPE too small\n");
        xcsf->pred->type = 0;
    } else {
        xcsf->pred->type = a;
    }
}

void
pred_param_set_type_string(struct XCSF *xcsf, const char *a)
{
    xcsf->pred->type = prediction_type_as_int(a);
}
