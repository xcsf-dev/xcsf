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
 * @date 2015--2021.
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
    param_set_n_actions(xcsf, n_actions);
    param_set_x_dim(xcsf, x_dim);
    param_set_y_dim(xcsf, y_dim);
    param_set_omp_num_threads(xcsf, 8);
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
const char *
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
    cJSON_AddItemToObject(json, "ea", cJSON_Parse(ea_param_json_export(xcsf)));
    switch (xcsf->cond->type) {
        case RULE_TYPE_DGP:
        case RULE_TYPE_NEURAL:
        case RULE_TYPE_NETWORK:
            break;
        default:
            if (xcsf->n_actions > 1) {
                cJSON *act_params = cJSON_Parse(action_param_json_export(xcsf));
                cJSON_AddItemToObject(json, "action", act_params);
            }
            break;
    }
    cJSON *cond_params = cJSON_Parse(cond_param_json_export(xcsf));
    cJSON_AddItemToObject(json, "condition", cond_params);
    cJSON *pred_params = cJSON_Parse(pred_param_json_export(xcsf));
    cJSON_AddItemToObject(json, "prediction", pred_params);
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}

/**
 * @brief Prints all XCSF parameters.
 * @param [in] xcsf The XCSF data structure.
 */
void
param_print(const struct XCSF *xcsf)
{
    printf("%s\n", param_json_export(xcsf));
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
 */
void
param_set_omp_num_threads(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set OMP_NUM_THREADS too small\n");
        xcsf->OMP_NUM_THREADS = 1;
    } else if (a > 1000) {
        printf("Warning: tried to set OMP_NUM_THREADS too large\n");
        xcsf->OMP_NUM_THREADS = 1000;
    } else {
        xcsf->OMP_NUM_THREADS = a;
    }
#ifdef PARALLEL
    omp_set_num_threads(xcsf->OMP_NUM_THREADS);
#endif
}

void
param_set_pop_init(struct XCSF *xcsf, const bool a)
{
    xcsf->POP_INIT = a;
}

void
param_set_max_trials(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set MAX_TRIALS too small\n");
        xcsf->MAX_TRIALS = 1;
    } else {
        xcsf->MAX_TRIALS = a;
    }
}

void
param_set_perf_trials(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set PERF_TRIALS too small\n");
        xcsf->PERF_TRIALS = 1;
    } else {
        xcsf->PERF_TRIALS = a;
    }
}

void
param_set_pop_size(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set POP_SIZE too small\n");
        xcsf->POP_SIZE = 1;
    } else {
        xcsf->POP_SIZE = a;
    }
}

void
param_set_loss_func_string(struct XCSF *xcsf, const char *a)
{
    xcsf->LOSS_FUNC = loss_type_as_int(a);
    loss_set_func(xcsf);
}

void
param_set_loss_func(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set LOSS_FUNC too small\n");
        xcsf->LOSS_FUNC = 0;
    } else if (a >= LOSS_NUM) {
        printf("Warning: tried to set LOSS_FUNC too large\n");
        xcsf->LOSS_FUNC = LOSS_NUM - 1;
    } else {
        xcsf->LOSS_FUNC = a;
    }
    loss_set_func(xcsf);
}

void
param_set_stateful(struct XCSF *xcsf, const bool a)
{
    xcsf->STATEFUL = a;
}

void
param_set_compaction(struct XCSF *xcsf, const bool a)
{
    xcsf->COMPACTION = a;
}

void
param_set_huber_delta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set HUBER_DELTA too small\n");
        xcsf->HUBER_DELTA = 0;
    } else {
        xcsf->HUBER_DELTA = a;
    }
}

void
param_set_gamma(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set GAMMA too small\n");
        xcsf->GAMMA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set GAMMA too large\n");
        xcsf->GAMMA = 1;
    } else {
        xcsf->GAMMA = a;
    }
}

void
param_set_teletransportation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set TELETRANSPORTATION too small\n");
        xcsf->TELETRANSPORTATION = 0;
    } else {
        xcsf->TELETRANSPORTATION = a;
    }
}

void
param_set_p_explore(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set P_EXPLORE too small\n");
        xcsf->P_EXPLORE = 0;
    } else if (a > 1) {
        printf("Warning: tried to set P_EXPLORE too large\n");
        xcsf->P_EXPLORE = 1;
    } else {
        xcsf->P_EXPLORE = a;
    }
}

void
param_set_alpha(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set ALPHA too small\n");
        xcsf->ALPHA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set ALPHA too large\n");
        xcsf->ALPHA = 1;
    } else {
        xcsf->ALPHA = a;
    }
}

void
param_set_beta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set BETA too small\n");
        xcsf->BETA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set BETA too large\n");
        xcsf->BETA = 1;
    } else {
        xcsf->BETA = a;
    }
}

void
param_set_delta(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set DELTA too small\n");
        xcsf->DELTA = 0;
    } else if (a > 1) {
        printf("Warning: tried to set DELTA too large\n");
        xcsf->DELTA = 1;
    } else {
        xcsf->DELTA = a;
    }
}

void
param_set_e0(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set E0 too small\n");
        xcsf->E0 = 0;
    } else {
        xcsf->E0 = a;
    }
}

void
param_set_init_error(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set INIT_ERROR too small\n");
        xcsf->INIT_ERROR = 0;
    } else {
        xcsf->INIT_ERROR = a;
    }
}

void
param_set_init_fitness(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set INIT_FITNESS too small\n");
        xcsf->INIT_FITNESS = 0;
    } else {
        xcsf->INIT_FITNESS = a;
    }
}

void
param_set_nu(struct XCSF *xcsf, const double a)
{
    if (a < 0) {
        printf("Warning: tried to set NU too small\n");
        xcsf->NU = 0;
    } else {
        xcsf->NU = a;
    }
}

void
param_set_theta_del(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set THETA_DEL too small\n");
        xcsf->THETA_DEL = 0;
    } else {
        xcsf->THETA_DEL = a;
    }
}

void
param_set_m_probation(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set M_PROBATION too small\n");
        xcsf->M_PROBATION = 0;
    } else {
        xcsf->M_PROBATION = a;
    }
}

void
param_set_set_subsumption(struct XCSF *xcsf, const bool a)
{
    xcsf->SET_SUBSUMPTION = a;
}

void
param_set_theta_sub(struct XCSF *xcsf, const int a)
{
    if (a < 0) {
        printf("Warning: tried to set THETA_SUB too small\n");
        xcsf->THETA_SUB = 0;
    } else {
        xcsf->THETA_SUB = a;
    }
}

void
param_set_x_dim(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set x_dim too small\n");
        xcsf->x_dim = 1;
    } else {
        xcsf->x_dim = a;
    }
}

void
param_set_explore(struct XCSF *xcsf, const bool a)
{
    xcsf->explore = a;
}

void
param_set_y_dim(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set y_dim too small\n");
        xcsf->y_dim = 1;
    } else {
        xcsf->y_dim = a;
    }
}

void
param_set_n_actions(struct XCSF *xcsf, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set n_actions too small\n");
        xcsf->n_actions = 1;
    } else {
        xcsf->n_actions = a;
    }
}
