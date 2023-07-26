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
 * @file param.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2023.
 * @brief Functions for setting and printing parameters.
 */

#include "param.h"
#include "action.h"
#include "condition.h"
#include "ea.h"
#include "prediction.h"
#include "utils.h"

#ifdef PARALLEL
    #include <omp.h>
#endif

#define MAX_LEN 512 //!< Maximum length of a population filename

/**
 * @brief Initialises default XCSF parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] x_dim The dimensionality of the input variables.
 * @param [in] y_dim The dimensionality of the prediction variables.
 * @param [in] n_actions The total number of possible actions.
 */
void
param_init(struct XCSF *xcsf, const int x_dim, const int y_dim,
           const int n_actions)
{
    xcsf->time = 0;
    xcsf->error = xcsf->E0;
    xcsf->mset_size = 0;
    xcsf->aset_size = 0;
    xcsf->mfrac = 0;
    xcsf->ea = malloc(sizeof(struct ArgsEA));
    xcsf->act = malloc(sizeof(struct ArgsAct));
    xcsf->cond = malloc(sizeof(struct ArgsCond));
    xcsf->pred = malloc(sizeof(struct ArgsPred));
    xcsf->population_file = malloc(sizeof(char));
    xcsf->population_file[0] = '\0';
    param_set_n_actions(xcsf, n_actions);
    param_set_x_dim(xcsf, x_dim);
    param_set_y_dim(xcsf, y_dim);
    param_set_omp_num_threads(xcsf, 8);
    param_set_random_state(xcsf, 0);
    param_set_pop_init(xcsf, true);
    param_set_max_trials(xcsf, 100000);
    param_set_perf_trials(xcsf, 1000);
    param_set_pop_size(xcsf, 2000);
    param_set_loss_func(xcsf, LOSS_MAE);
    param_set_huber_delta(xcsf, 1);
    param_set_e0(xcsf, 0.01);
    param_set_alpha(xcsf, 0.1);
    param_set_nu(xcsf, 5);
    param_set_beta(xcsf, 0.1);
    param_set_delta(xcsf, 0.1);
    param_set_theta_del(xcsf, 20);
    param_set_init_fitness(xcsf, 0.01);
    param_set_init_error(xcsf, 0);
    param_set_m_probation(xcsf, 10000);
    param_set_stateful(xcsf, true);
    param_set_compaction(xcsf, false);
    param_set_gamma(xcsf, 0.95);
    param_set_teletransportation(xcsf, 50);
    param_set_p_explore(xcsf, 0.9);
    param_set_set_subsumption(xcsf, false);
    param_set_theta_sub(xcsf, 100);
    ea_param_defaults(xcsf);
    action_param_defaults(xcsf);
    cond_param_defaults(xcsf);
    pred_param_defaults(xcsf);
}

void
param_free(struct XCSF *xcsf)
{
    if (xcsf->population_file != NULL) {
        free(xcsf->population_file);
    }
    action_param_free(xcsf);
    cond_param_free(xcsf);
    pred_param_free(xcsf);
    free(xcsf->ea);
    free(xcsf->act);
    free(xcsf->cond);
    free(xcsf->pred);
}

/**
 * @brief Returns a json formatted string representation of the parameters.
 * @param [in] xcsf XCSF data structure.
 * @return String encoded in json format.
 */
char *
param_json_export(const struct XCSF *xcsf)
{
    char v[256];
    snprintf(v, 256, "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD);
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "version", v);
    cJSON_AddNumberToObject(json, "x_dim", xcsf->x_dim);
    cJSON_AddNumberToObject(json, "y_dim", xcsf->y_dim);
    cJSON_AddNumberToObject(json, "n_actions", xcsf->n_actions);
    cJSON_AddNumberToObject(json, "omp_num_threads", xcsf->OMP_NUM_THREADS);
    cJSON_AddNumberToObject(json, "random_state", xcsf->RANDOM_STATE);
    if (xcsf->population_file != NULL) {
        cJSON_AddStringToObject(json, "population_file", xcsf->population_file);
    }
    cJSON_AddBoolToObject(json, "pop_init", xcsf->POP_INIT);
    cJSON_AddNumberToObject(json, "max_trials", xcsf->MAX_TRIALS);
    cJSON_AddNumberToObject(json, "perf_trials", xcsf->PERF_TRIALS);
    cJSON_AddNumberToObject(json, "pop_size", xcsf->POP_SIZE);
    cJSON_AddStringToObject(json, "loss_func",
                            loss_type_as_string(xcsf->LOSS_FUNC));
    if (xcsf->LOSS_FUNC == LOSS_HUBER) {
        cJSON_AddNumberToObject(json, "huber_delta", xcsf->HUBER_DELTA);
    }
    if (xcsf->n_actions > 1) {
        cJSON_AddNumberToObject(json, "gamma", xcsf->GAMMA);
        cJSON_AddNumberToObject(json, "teletransportation",
                                xcsf->TELETRANSPORTATION);
        cJSON_AddNumberToObject(json, "p_explore", xcsf->P_EXPLORE);
    }
    cJSON_AddBoolToObject(json, "set_subsumption", xcsf->SET_SUBSUMPTION);
    cJSON_AddNumberToObject(json, "theta_sub", xcsf->THETA_SUB);
    cJSON_AddNumberToObject(json, "e0", xcsf->E0);
    cJSON_AddNumberToObject(json, "alpha", xcsf->ALPHA);
    cJSON_AddNumberToObject(json, "nu", xcsf->NU);
    cJSON_AddNumberToObject(json, "beta", xcsf->BETA);
    cJSON_AddNumberToObject(json, "delta", xcsf->DELTA);
    cJSON_AddNumberToObject(json, "theta_del", xcsf->THETA_DEL);
    cJSON_AddNumberToObject(json, "init_fitness", xcsf->INIT_FITNESS);
    cJSON_AddNumberToObject(json, "init_error", xcsf->INIT_ERROR);
    cJSON_AddNumberToObject(json, "m_probation", xcsf->M_PROBATION);
    cJSON_AddBoolToObject(json, "stateful", xcsf->STATEFUL);
    cJSON_AddBoolToObject(json, "compaction", xcsf->COMPACTION);
    char *ea_param_str = ea_param_json_export(xcsf);
    cJSON *ea_params = cJSON_Parse(ea_param_str);
    cJSON_AddItemToObject(json, "ea", ea_params);
    free(ea_param_str);
    switch (xcsf->cond->type) {
        case RULE_TYPE_DGP:
        case RULE_TYPE_NEURAL:
        case RULE_TYPE_NETWORK:
            break;
        default:
            if (xcsf->n_actions > 1) {
                char *act_param_str = action_param_json_export(xcsf);
                cJSON *act_params = cJSON_Parse(act_param_str);
                cJSON_AddItemToObject(json, "action", act_params);
                free(act_param_str);
            }
            break;
    }
    char *cond_param_str = cond_param_json_export(xcsf);
    cJSON *cond_params = cJSON_Parse(cond_param_str);
    cJSON_AddItemToObject(json, "condition", cond_params);
    free(cond_param_str);
    char *pred_param_str = pred_param_json_export(xcsf);
    cJSON *pred_params = cJSON_Parse(pred_param_str);
    cJSON_AddItemToObject(json, "prediction", pred_params);
    free(pred_param_str);
    char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Sets the general parameters from a cJSON object.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 * @return Whether a parameter was found.
 */
static bool
param_json_import_general(struct XCSF *xcsf, const cJSON *json)
{
    if (strncmp(json->string, "version\0", 8) == 0) {
        return true;
    } else if (strncmp(json->string, "x_dim\0", 6) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_x_dim(xcsf, json->valueint));
    } else if (strncmp(json->string, "y_dim\0", 6) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_y_dim(xcsf, json->valueint));
    } else if (strncmp(json->string, "n_actions\0", 10) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_n_actions(xcsf, json->valueint));
    } else if (strncmp(json->string, "omp_num_threads\0", 16) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_omp_num_threads(xcsf, json->valueint));
    } else if (strncmp(json->string, "random_state\0", 13) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_random_state(xcsf, json->valueint));
    } else if (strncmp(json->string, "population_file\0", 16) == 0 &&
               cJSON_IsString(json)) {
        catch_error(param_set_population_file(xcsf, json->valuestring));
    } else if (strncmp(json->string, "pop_size\0", 9) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_pop_size(xcsf, json->valueint));
    } else if (strncmp(json->string, "max_trials\0", 11) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_max_trials(xcsf, json->valueint));
    } else if (strncmp(json->string, "pop_init\0", 9) == 0 &&
               cJSON_IsBool(json)) {
        const bool init = true ? json->type == cJSON_True : false;
        catch_error(param_set_pop_init(xcsf, init));
    } else if (strncmp(json->string, "perf_trials\0", 12) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_perf_trials(xcsf, json->valueint));
    } else if (strncmp(json->string, "loss_func\0", 10) == 0 &&
               cJSON_IsString(json)) {
        if (param_set_loss_func_string(xcsf, json->valuestring) ==
            PARAM_INVALID) {
            printf("Invalid loss function: %s\n", json->valuestring);
            printf("Options: {%s}\n", LOSS_OPTIONS);
            exit(EXIT_FAILURE);
        }
    } else if (strncmp(json->string, "huber_delta\0", 12) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_huber_delta(xcsf, json->valuedouble));
    } else {
        return false;
    }
    return true;
}

/**
 * @brief Sets the multi-step parameters from a cJSON object.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 * @return Whether a parameter was found.
 *
 */
static bool
param_json_import_multi(struct XCSF *xcsf, const cJSON *json)
{
    if (strncmp(json->string, "teletransportation\0", 19) == 0 &&
        cJSON_IsNumber(json)) {
        catch_error(param_set_teletransportation(xcsf, json->valueint));
    } else if (strncmp(json->string, "gamma\0", 6) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_gamma(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "p_explore\0", 10) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_p_explore(xcsf, json->valuedouble));
    } else {
        return false;
    }
    return true;
}

/**
 * @brief Sets the subsumption parameters from a cJSON object.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 * @return Whether a parameter was found.
 *
 */
static bool
param_json_import_subsump(struct XCSF *xcsf, const cJSON *json)
{
    if (strncmp(json->string, "set_subsumption\0", 16) == 0 &&
        cJSON_IsBool(json)) {
        const bool sub = true ? json->type == cJSON_True : false;
        catch_error(param_set_set_subsumption(xcsf, sub));
    } else if (strncmp(json->string, "theta_sub\0", 10) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_theta_sub(xcsf, json->valueint));
    } else {
        return false;
    }
    return true;
}

/**
 * @brief Sets the general classifier parameters from a cJSON object.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 * @return Whether a parameter was found.
 *
 */
static bool
param_json_import_cl_general(struct XCSF *xcsf, const cJSON *json)
{
    if (strncmp(json->string, "alpha\0", 6) == 0 && cJSON_IsNumber(json)) {
        catch_error(param_set_alpha(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "beta\0", 5) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_beta(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "delta\0", 6) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_delta(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "nu\0", 3) == 0 && cJSON_IsNumber(json)) {
        catch_error(param_set_nu(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "theta_del\0", 10) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_theta_del(xcsf, json->valueint));
    } else if (strncmp(json->string, "init_fitness\0", 13) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_init_fitness(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "init_error\0", 11) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_init_error(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "e0\0", 3) == 0 && cJSON_IsNumber(json)) {
        catch_error(param_set_e0(xcsf, json->valuedouble));
    } else if (strncmp(json->string, "m_probation\0", 12) == 0 &&
               cJSON_IsNumber(json)) {
        catch_error(param_set_m_probation(xcsf, json->valueint));
    } else if (strncmp(json->string, "stateful\0", 9) == 0 &&
               cJSON_IsBool(json)) {
        const bool stateful = true ? json->type == cJSON_True : false;
        catch_error(param_set_stateful(xcsf, stateful));
    } else if (strncmp(json->string, "compaction\0", 11) == 0 &&
               cJSON_IsBool(json)) {
        const bool compact = true ? json->type == cJSON_True : false;
        catch_error(param_set_compaction(xcsf, compact));
    } else {
        return false;
    }
    return true;
}

/**
 * @brief Sets the action parameters from a json formatted string.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 */
static void
param_json_import_action(struct XCSF *xcsf, cJSON *json)
{
    if (strncmp(json->string, "type\0", 5) == 0 && cJSON_IsString(json) &&
        action_param_set_type_string(xcsf, json->valuestring) ==
            ACT_TYPE_INVALID) {
        printf("Invalid action type: %s\n", json->valuestring);
        printf("Options: {%s}\n", ACT_TYPE_OPTIONS);
        exit(EXIT_FAILURE);
    }
    json = json->next;
    if (json != NULL && strncmp(json->string, "args\0", 5) == 0) {
        const char *ret = action_param_json_import(xcsf, json);
        if (ret != NULL) {
            printf("Invalid action parameter %s\n", ret);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Sets the condition parameters from a json formatted string.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 */
static void
param_json_import_condition(struct XCSF *xcsf, cJSON *json)
{
    if (strncmp(json->string, "type\0", 5) == 0 && cJSON_IsString(json) &&
        cond_param_set_type_string(xcsf, json->valuestring) ==
            COND_TYPE_INVALID) {
        printf("Invalid condition type: %s\n", json->valuestring);
        printf("Options: {%s}\n", COND_TYPE_OPTIONS);
        exit(EXIT_FAILURE);
    }
    json = json->next;
    if (json != NULL && strncmp(json->string, "args\0", 5) == 0) {
        const char *ret = cond_param_json_import(xcsf, json);
        if (ret != NULL) {
            printf("Invalid condition parameter %s\n", ret);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Sets the prediction parameters from a json formatted string.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json cJSON object.
 */
static void
param_json_import_prediction(struct XCSF *xcsf, cJSON *json)
{
    if (strncmp(json->string, "type\0", 5) == 0 && cJSON_IsString(json) &&
        pred_param_set_type_string(xcsf, json->valuestring) ==
            PRED_TYPE_INVALID) {
        printf("Invalid prediction type: %s\n", json->valuestring);
        printf("Options: {%s}\n", PRED_TYPE_OPTIONS);
        exit(EXIT_FAILURE);
    }
    json = json->next;
    if (json != NULL && strncmp(json->string, "args\0", 5) == 0) {
        const char *ret = pred_param_json_import(xcsf, json);
        if (ret != NULL) {
            printf("Invalid prediction parameter %s\n", ret);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Sets the parameters from a json formatted string.
 * @param [in,out] xcsf The XCSF data structure.
 * @param [in] json_str String encoded in json format.
 */
void
param_json_import(struct XCSF *xcsf, const char *json_str)
{
    cJSON *json = cJSON_Parse(json_str);
    utils_json_parse_check(json);
    for (cJSON *iter = json->child; iter != NULL; iter = iter->next) {
        if (param_json_import_general(xcsf, iter)) {
            continue;
        }
        if (param_json_import_multi(xcsf, iter)) {
            continue;
        }
        if (param_json_import_subsump(xcsf, iter)) {
            continue;
        }
        if (param_json_import_cl_general(xcsf, iter)) {
            continue;
        }
        if (strncmp(iter->string, "ea\0", 3) == 0) {
            ea_param_json_import(xcsf, iter->child);
            continue;
        }
        if (strncmp(iter->string, "action\0", 7) == 0) {
            param_json_import_action(xcsf, iter->child);
            continue;
        }
        if (strncmp(iter->string, "condition\0", 10) == 0) {
            param_json_import_condition(xcsf, iter->child);
            continue;
        }
        if (strncmp(iter->string, "prediction\0", 11) == 0) {
            param_json_import_prediction(xcsf, iter->child);
            continue;
        }
        printf("Error: unable to import parameter: %s\n", iter->string);
        exit(EXIT_FAILURE);
    }
    cJSON_Delete(json);
}

/**
 * @brief Prints all XCSF parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
param_print(const struct XCSF *xcsf)
{
    char *json_str = param_json_export(xcsf);
    printf("%s\n", json_str);
    free(json_str);
}

/**
 * @brief Writes the XCSF data structure to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the output file.
 * @return The total number of elements written.
 */
size_t
param_save(const struct XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    size_t len = strnlen(xcsf->population_file, MAX_LEN);
    s += fwrite(&len, sizeof(size_t), 1, fp);
    s += fwrite(xcsf->population_file, sizeof(char), len, fp);
    s += fwrite(&xcsf->time, sizeof(int), 1, fp);
    s += fwrite(&xcsf->error, sizeof(double), 1, fp);
    s += fwrite(&xcsf->mset_size, sizeof(double), 1, fp);
    s += fwrite(&xcsf->aset_size, sizeof(double), 1, fp);
    s += fwrite(&xcsf->mfrac, sizeof(double), 1, fp);
    s += fwrite(&xcsf->explore, sizeof(bool), 1, fp);
    s += fwrite(&xcsf->x_dim, sizeof(int), 1, fp);
    s += fwrite(&xcsf->y_dim, sizeof(int), 1, fp);
    s += fwrite(&xcsf->n_actions, sizeof(int), 1, fp);
    s += fwrite(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->RANDOM_STATE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->POP_INIT, sizeof(bool), 1, fp);
    s += fwrite(&xcsf->MAX_TRIALS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->PERF_TRIALS, sizeof(int), 1, fp);
    s += fwrite(&xcsf->POP_SIZE, sizeof(int), 1, fp);
    s += fwrite(&xcsf->LOSS_FUNC, sizeof(int), 1, fp);
    s += fwrite(&xcsf->HUBER_DELTA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->GAMMA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->TELETRANSPORTATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->P_EXPLORE, sizeof(double), 1, fp);
    s += fwrite(&xcsf->SET_SUBSUMPTION, sizeof(bool), 1, fp);
    s += fwrite(&xcsf->THETA_SUB, sizeof(int), 1, fp);
    s += fwrite(&xcsf->E0, sizeof(double), 1, fp);
    s += fwrite(&xcsf->ALPHA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->NU, sizeof(double), 1, fp);
    s += fwrite(&xcsf->BETA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->DELTA, sizeof(double), 1, fp);
    s += fwrite(&xcsf->THETA_DEL, sizeof(int), 1, fp);
    s += fwrite(&xcsf->INIT_FITNESS, sizeof(double), 1, fp);
    s += fwrite(&xcsf->INIT_ERROR, sizeof(double), 1, fp);
    s += fwrite(&xcsf->M_PROBATION, sizeof(int), 1, fp);
    s += fwrite(&xcsf->STATEFUL, sizeof(bool), 1, fp);
    s += fwrite(&xcsf->COMPACTION, sizeof(bool), 1, fp);
    s += ea_param_save(xcsf, fp);
    s += action_param_save(xcsf, fp);
    s += cond_param_save(xcsf, fp);
    s += pred_param_save(xcsf, fp);
    return s;
}

/**
 * @brief Reads the XCSF data structure from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the input file.
 * @return The total number of elements read.
 */
size_t
param_load(struct XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    size_t len = 0;
    s += fread(&len, sizeof(size_t), 1, fp);
    free(xcsf->population_file);
    xcsf->population_file = malloc(sizeof(char) * len);
    s += fread(xcsf->population_file, sizeof(char), len, fp);
    s += fread(&xcsf->time, sizeof(int), 1, fp);
    s += fread(&xcsf->error, sizeof(double), 1, fp);
    s += fread(&xcsf->mset_size, sizeof(double), 1, fp);
    s += fread(&xcsf->aset_size, sizeof(double), 1, fp);
    s += fread(&xcsf->mfrac, sizeof(double), 1, fp);
    s += fread(&xcsf->explore, sizeof(bool), 1, fp);
    s += fread(&xcsf->x_dim, sizeof(int), 1, fp);
    s += fread(&xcsf->y_dim, sizeof(int), 1, fp);
    s += fread(&xcsf->n_actions, sizeof(int), 1, fp);
    if (xcsf->x_dim < 1 || xcsf->y_dim < 1 || xcsf->n_actions < 1) {
        printf("param_load(): read error\n");
        exit(EXIT_FAILURE);
    }
    s += fread(&xcsf->OMP_NUM_THREADS, sizeof(int), 1, fp);
    s += fread(&xcsf->RANDOM_STATE, sizeof(int), 1, fp);
    s += fread(&xcsf->POP_INIT, sizeof(bool), 1, fp);
    s += fread(&xcsf->MAX_TRIALS, sizeof(int), 1, fp);
    s += fread(&xcsf->PERF_TRIALS, sizeof(int), 1, fp);
    s += fread(&xcsf->POP_SIZE, sizeof(int), 1, fp);
    s += fread(&xcsf->LOSS_FUNC, sizeof(int), 1, fp);
    s += fread(&xcsf->HUBER_DELTA, sizeof(double), 1, fp);
    s += fread(&xcsf->GAMMA, sizeof(double), 1, fp);
    s += fread(&xcsf->TELETRANSPORTATION, sizeof(int), 1, fp);
    s += fread(&xcsf->P_EXPLORE, sizeof(double), 1, fp);
    s += fread(&xcsf->SET_SUBSUMPTION, sizeof(bool), 1, fp);
    s += fread(&xcsf->THETA_SUB, sizeof(int), 1, fp);
    s += fread(&xcsf->E0, sizeof(double), 1, fp);
    s += fread(&xcsf->ALPHA, sizeof(double), 1, fp);
    s += fread(&xcsf->NU, sizeof(double), 1, fp);
    s += fread(&xcsf->BETA, sizeof(double), 1, fp);
    s += fread(&xcsf->DELTA, sizeof(double), 1, fp);
    s += fread(&xcsf->THETA_DEL, sizeof(int), 1, fp);
    s += fread(&xcsf->INIT_FITNESS, sizeof(double), 1, fp);
    s += fread(&xcsf->INIT_ERROR, sizeof(double), 1, fp);
    s += fread(&xcsf->M_PROBATION, sizeof(int), 1, fp);
    s += fread(&xcsf->STATEFUL, sizeof(bool), 1, fp);
    s += fread(&xcsf->COMPACTION, sizeof(bool), 1, fp);
    s += ea_param_load(xcsf, fp);
    s += action_param_load(xcsf, fp);
    s += cond_param_load(xcsf, fp);
    s += pred_param_load(xcsf, fp);
    loss_set_func(xcsf);
    return s;
}

/* setters */

/**
 * @brief Sets the number of OMP threads.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] a The number of threads.
 * @return NULL if successful; or an error message.
 */
const char *
param_set_omp_num_threads(struct XCSF *xcsf, const int a)
{
    if (a < 1 || a > 1000) {
        return "Invalid OMP_NUM_THREADS. Range: [1,1000]";
    }
    xcsf->OMP_NUM_THREADS = a;
#ifdef PARALLEL
    omp_set_num_threads(xcsf->OMP_NUM_THREADS);
#endif
    return NULL;
}

const char *
param_set_random_state(struct XCSF *xcsf, const int a)
{
    xcsf->RANDOM_STATE = a;
    if (a > 0) {
        rand_init_seed(a);
    } else {
        rand_init();
    }
    return NULL;
}

const char *
param_set_population_file(struct XCSF *xcsf, const char *a)
{
    free(xcsf->population_file);
    size_t length = strnlen(a, sizeof(char) * MAX_LEN) + 1;
    xcsf->population_file = malloc(sizeof(char) * length);
    strncpy(xcsf->population_file, a, length);
    return NULL;
}

const char *
param_set_pop_init(struct XCSF *xcsf, const bool a)
{
    xcsf->POP_INIT = a;
    return NULL;
}

const char *
param_set_max_trials(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        return "MAX_TRIALS must be > 0";
    }
    xcsf->MAX_TRIALS = a;
    return NULL;
}

const char *
param_set_perf_trials(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        return "PERF_TRIALS must be > 0";
    }
    xcsf->PERF_TRIALS = a;
    return NULL;
}

const char *
param_set_pop_size(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        return "POP_SIZE must be > 0";
    }
    xcsf->POP_SIZE = a;
    return NULL;
}

int
param_set_loss_func_string(struct XCSF *xcsf, const char *a)
{
    const int loss = loss_type_as_int(a);
    if (loss != LOSS_INVALID) {
        xcsf->LOSS_FUNC = loss;
        if (loss_set_func(xcsf) != LOSS_INVALID) {
            return PARAM_FOUND;
        }
    }
    return PARAM_INVALID;
}

void
param_set_loss_func(struct XCSF *xcsf, const int a)
{
    if (a < 0 || a >= LOSS_NUM) {
        printf("param_set_loss_func(): invalid LOSS_FUNC: %d\n", a);
        exit(EXIT_FAILURE);
    }
    xcsf->LOSS_FUNC = a;
    loss_set_func(xcsf);
}

const char *
param_set_stateful(struct XCSF *xcsf, const bool a)
{
    xcsf->STATEFUL = a;
    return NULL;
}

const char *
param_set_compaction(struct XCSF *xcsf, const bool a)
{
    xcsf->COMPACTION = a;
    return NULL;
}

const char *
param_set_huber_delta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        return "HUBER_DELTA must be >= 0";
    }
    xcsf->HUBER_DELTA = a;
    return NULL;
}

const char *
param_set_gamma(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid GAMMA. Range: [0,1]";
    }
    xcsf->GAMMA = a;
    return NULL;
}

const char *
param_set_teletransportation(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        return "TELETRANSPORTATION must be > 0";
    }
    xcsf->TELETRANSPORTATION = a;
    return NULL;
}

const char *
param_set_p_explore(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid P_EXPLORE. Range: [0,1]";
    }
    xcsf->P_EXPLORE = a;
    return NULL;
}

const char *
param_set_alpha(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid ALPHA. Range: [0,1]";
    }
    xcsf->ALPHA = a;
    return NULL;
}

const char *
param_set_beta(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid BETA. Range: [0,1]";
    }
    xcsf->BETA = a;
    return NULL;
}

const char *
param_set_delta(struct XCSF *xcsf, const double a)
{
    if (a < 0 || a > 1) {
        return "Invalid DELTA. Range: [0,1]";
    }
    xcsf->DELTA = a;
    return NULL;
}

const char *
param_set_e0(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        return "E0 must be >= 0";
    }
    xcsf->E0 = a;
    return NULL;
}

const char *
param_set_init_error(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        return "INIT_ERROR must be >= 0";
    }
    xcsf->INIT_ERROR = a;
    return NULL;
}

const char *
param_set_init_fitness(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        return "INIT_FITNESS must be >= 0";
    }
    xcsf->INIT_FITNESS = a;
    return NULL;
}

const char *
param_set_nu(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        return "NU must be >= 0";
    }
    xcsf->NU = a;
    return NULL;
}

const char *
param_set_theta_del(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        return "THETA_DEL must be >= 0";
    }
    xcsf->THETA_DEL = a;
    return NULL;
}

const char *
param_set_m_probation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        return "M_PROBATION must be >= 0";
    }
    xcsf->M_PROBATION = a;
    return NULL;
}

const char *
param_set_set_subsumption(struct XCSF *xcsf, const bool a)
{
    xcsf->SET_SUBSUMPTION = a;
    return NULL;
}

const char *
param_set_theta_sub(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        return "THETA_SUB must be >= 0";
    }
    xcsf->THETA_SUB = a;
    return NULL;
}

const char *
param_set_x_dim(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        return "x_dim must be > 0";
    }
    xcsf->x_dim = a;
    return NULL;
}

const char *
param_set_explore(struct XCSF *xcsf, const bool a)
{
    xcsf->explore = a;
    return NULL;
}

const char *
param_set_y_dim(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        return "y_dim must be > 0";
    }
    xcsf->y_dim = a;
    return NULL;
}

const char *
param_set_n_actions(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        return "n_actions must be > 0";
    }
    xcsf->n_actions = a;
    return NULL;
}
