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
 * @file config.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Configuration file handling functions.
 */ 

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <errno.h>
#include "xcsf.h"
#include "param.h"
#include "config.h"

#define ARRAY_DELIM "," //!< Delimeter for config arrays
#define MAXLEN 127 //!< Maximum config file line length to read
#define BASE 10 //!< Decimal numbers

static void config_add_param(XCSF *xcsf, const char *name, char *value);
static void config_get_ints(char *str, int *val);
static void config_newnvpair(XCSF *xcsf, const char *param);
static void config_process(XCSF *xcsf, const char *configline);
static void config_trim(char *s);
static void config_cl_action(XCSF *xcsf, const char *name, const char *value);
static void config_cl_condition(XCSF *xcsf, const char *name, char *value);
static void config_cl_general(XCSF *xcsf, const char *name, const char *value);
static void config_cl_prediction(XCSF *xcsf, const char *name, char *value);
static void config_ea(XCSF *xcsf, const char *name, const char *value);
static void config_general(XCSF *xcsf, const char *name, const char *value);
static void config_multistep(XCSF *xcsf, const char *name, const char *value);
static void config_subsumption(XCSF *xcsf, const char *name, const char *value);

/**
 * @brief Reads the specified configuration file.
 * @param xcsf The XCSF data structure.
 * @param filename The name of the configuration file.
 */
void config_read(XCSF *xcsf, const char *filename) {
    FILE *f = fopen(filename, "rt");
    if(f == NULL) {
        printf("Warning: could not open %s.\n", filename);
        return;
    }
    char buff[MAXLEN];
    while(!feof(f)) {
        if(fgets(buff, MAXLEN-2, f) == NULL) {
            break;
        }
        config_trim(buff);
        config_process(xcsf, buff);
    }
    fclose(f);
}

/**
 * @brief Sets specified parameter.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_add_param(XCSF *xcsf, const char *name, char *value)
{
    config_general(xcsf, name, value);
    config_multistep(xcsf, name, value);
    config_subsumption(xcsf, name, value);
    config_ea(xcsf, name, value);
    config_cl_general(xcsf, name, value);
    config_cl_condition(xcsf, name, value);
    config_cl_prediction(xcsf, name, value);
    config_cl_action(xcsf, name, value);
}

/**
 * @brief Sets general XCSF parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_general(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "OMP_NUM_THREADS", 16) == 0) {
        param_set_omp_num_threads(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "POP_SIZE", 9) == 0) {
        param_set_pop_size(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "MAX_TRIALS", 10) == 0) {
        param_set_max_trials(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "POP_INIT", 9) == 0) {
        param_set_pop_init(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_pop_init(xcsf, true);
        }
    }
    else if(strncmp(name, "PERF_TRIALS", 12) == 0) {
        param_set_perf_trials(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "LOSS_FUNC", 10) == 0) {
        param_set_loss_func(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "AUTO_ENCODE", 9) == 0) {
        param_set_auto_encode(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_auto_encode(xcsf, true);
        }
    }
}

/**
 * @brief Sets multistep experiment parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_multistep(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "TELETRANSPORTATION", 19) == 0) {
        param_set_teletransportation(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "GAMMA", 6) == 0) {
        param_set_gamma(xcsf, atof(value));
    }
    else if(strncmp(name, "P_EXPLORE", 10) == 0) {
        param_set_p_explore(xcsf, atof(value));
    }
}

/**
 * @brief Sets subsumption parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_subsumption(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "EA_SUBSUMPTION", 15) == 0) {
        param_set_ea_subsumption(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_ea_subsumption(xcsf, true);
        }
    }
    else if(strncmp(name, "SET_SUBSUMPTION", 16) == 0) {
        param_set_set_subsumption(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_set_subsumption(xcsf, true);
        }
    }
    else if(strncmp(name, "THETA_SUB", 10) == 0) {
        param_set_theta_sub(xcsf, strtoimax(value, &end, BASE));
    }
}

/**
 * @brief Sets evolutionary algorithm parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_ea(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "EA_SELECT_TYPE", 15) == 0) {
        param_set_ea_select_type(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "EA_SELECT_SIZE", 15) == 0) {
        param_set_ea_select_size(xcsf, atof(value));
    }
    else if(strncmp(name, "THETA_EA", 9) == 0) {
        param_set_theta_ea(xcsf, atof(value));
    }
    else if(strncmp(name, "LAMBDA", 7) == 0) {
        param_set_lambda(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "P_CROSSOVER", 12) == 0) {
        param_set_p_crossover(xcsf, atof(value));
    }
    else if(strncmp(name, "SAM_TYPE", 9) == 0) {
        param_set_sam_type(xcsf, strtoimax(value, &end, BASE));
    }
}

/**
 * @brief Sets general classifier parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_cl_general(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "ALPHA", 6) == 0) {
        param_set_alpha(xcsf, atof(value));
    }
    else if(strncmp(name, "BETA", 5) == 0) {
        param_set_beta(xcsf, atof(value));
    }
    else if(strncmp(name, "DELTA", 6) == 0) {
        param_set_delta(xcsf, atof(value));
    }
    else if(strncmp(name, "NU", 3) == 0) {
        param_set_nu(xcsf, atof(value));
    }
    else if(strncmp(name, "THETA_DEL", 10) == 0) {
        param_set_theta_del(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "INIT_FITNESS", 13) == 0) {
        param_set_init_fitness(xcsf, atof(value));
    }
    else if(strncmp(name, "INIT_ERROR", 11) == 0) {
        param_set_init_error(xcsf, atof(value));
    }
    else if(strncmp(name, "ERR_REDUC", 10) == 0) {
        param_set_err_reduc(xcsf, atof(value));
    }
    else if(strncmp(name, "FIT_REDUC", 10) == 0) {
        param_set_fit_reduc(xcsf, atof(value));
    }
    else if(strncmp(name, "EPS_0", 6) == 0) {
        param_set_eps_0(xcsf, atof(value));
    }
    else if(strncmp(name, "M_PROBATION", 12) == 0) {
        param_set_m_probation(xcsf, strtoimax(value, &end, BASE));
    }
}

/**
 * @brief Sets classifier condition parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_cl_condition(XCSF *xcsf, const char *name, char *value)
{
    char *end;
    if(strncmp(name, "COND_MIN", 9) == 0) {
        param_set_cond_min(xcsf, atof(value));
    }
    else if(strncmp(name, "COND_MAX", 9) == 0) {
        param_set_cond_max(xcsf, atof(value));
    }
    else if(strncmp(name, "COND_TYPE", 10) == 0) {
        param_set_cond_type(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "COND_SMIN", 10) == 0) {
        param_set_cond_smin(xcsf, atof(value));
    }
    else if(strncmp(name, "COND_BITS", 10) == 0) {
        param_set_cond_bits(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "COND_ETA", 9) == 0) {
        param_set_cond_eta(xcsf, atof(value));
    }
    else if(strncmp(name, "GP_NUM_CONS", 12) == 0) {
        param_set_gp_num_cons(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "GP_INIT_DEPTH", 14) == 0) {
        param_set_gp_init_depth(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "DGP_NUM_NODES", 14) == 0) {
        param_set_dgp_num_nodes(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "RESET_STATES", 13) == 0) {
        param_set_reset_states(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_reset_states(xcsf, true);
        }
    }
    else if(strncmp(name, "MAX_K", 6) == 0) {
        param_set_max_k(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "MAX_T", 6) == 0) {
        param_set_max_t(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "MAX_NEURON_MOD", 15) == 0) {
        param_set_max_neuron_mod(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "COND_EVOLVE_WEIGHTS", 20) == 0) {
        param_set_cond_evolve_weights(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_cond_evolve_weights(xcsf, true);
        }
    }
    else if(strncmp(name, "COND_EVOLVE_NEURONS", 20) == 0) {
        param_set_cond_evolve_neurons(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_cond_evolve_neurons(xcsf, true);
        }
    }
    else if(strncmp(name, "COND_EVOLVE_FUNCTIONS", 22) == 0) {
        param_set_cond_evolve_functions(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_cond_evolve_functions(xcsf, true);
        }
    }
    else if(strncmp(name, "COND_NUM_NEURONS", 17) == 0) {
        memset(xcsf->COND_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
        config_get_ints(value, xcsf->COND_NUM_NEURONS);
    }
    else if(strncmp(name, "COND_MAX_NEURONS", 17) == 0) {
        memset(xcsf->COND_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
        config_get_ints(value, xcsf->COND_MAX_NEURONS);
    }
    else if(strncmp(name, "COND_OUTPUT_ACTIVATION", 23) == 0) {
        param_set_cond_output_activation(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "COND_HIDDEN_ACTIVATION", 23) == 0) {
        param_set_cond_hidden_activation(xcsf, strtoimax(value, &end, BASE));
    }
}

/**
 * @brief Sets classifier prediction parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_cl_prediction(XCSF *xcsf, const char *name, char *value)
{
    char *end;
    if(strncmp(name, "PRED_TYPE", 10) == 0) {
        param_set_pred_type(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "PRED_ETA", 9) == 0) {
        param_set_pred_eta(xcsf, atof(value));
    }
    else if(strncmp(name, "PRED_RESET", 11) == 0) {
        param_set_pred_reset(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_pred_reset(xcsf, true);
        }
    }
    else if(strncmp(name, "PRED_X0", 8) == 0) {
        param_set_pred_x0(xcsf, atof(value));
    }
    else if(strncmp(name, "PRED_RLS_SCALE_FACTOR", 22) == 0) {
        param_set_pred_rls_scale_factor(xcsf, atof(value));
    }
    else if(strncmp(name, "PRED_RLS_LAMBDA", 16) == 0) {
        param_set_pred_rls_lambda(xcsf, atof(value));
    }
    else if(strncmp(name, "PRED_EVOLVE_WEIGHTS", 20) == 0) {
        param_set_pred_evolve_weights(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_pred_evolve_weights(xcsf, true);
        }
    }
    else if(strncmp(name, "PRED_EVOLVE_NEURONS", 20) == 0) {
        param_set_pred_evolve_neurons(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_pred_evolve_neurons(xcsf, true);
        }
    }
    else if(strncmp(name, "PRED_EVOLVE_FUNCTIONS", 22) == 0) {
        param_set_pred_evolve_functions(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_pred_evolve_functions(xcsf, true);
        }
    }
    else if(strncmp(name, "PRED_EVOLVE_ETA", 16) == 0) {
        param_set_pred_evolve_eta(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_pred_evolve_eta(xcsf, true);
        }
    }
    else if(strncmp(name, "PRED_SGD_WEIGHTS", 17) == 0) {
        param_set_pred_sgd_weights(xcsf, false);
        if(strncmp(value, "true", 5) == 0) {
            param_set_pred_sgd_weights(xcsf, true);
        }
    }
    else if(strncmp(name, "PRED_MOMENTUM", 14) == 0) {
        param_set_pred_momentum(xcsf, atof(value));
    }
    else if(strncmp(name, "PRED_NUM_NEURONS", 17) == 0) {
        memset(xcsf->PRED_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
        config_get_ints(value, xcsf->PRED_NUM_NEURONS);
    }
    else if(strncmp(name, "PRED_MAX_NEURONS", 17) == 0) {
        memset(xcsf->PRED_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
        config_get_ints(value, xcsf->PRED_MAX_NEURONS);
    }
    else if(strncmp(name, "PRED_OUTPUT_ACTIVATION", 23) == 0) {
        param_set_pred_output_activation(xcsf, strtoimax(value, &end, BASE));
    }
    else if(strncmp(name, "PRED_HIDDEN_ACTIVATION", 23) == 0) {
        param_set_pred_hidden_activation(xcsf, strtoimax(value, &end, BASE));
    }
}

/**
 * @brief Sets classifier action parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void config_cl_action(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "ACT_TYPE", 9) == 0) {
        param_set_act_type(xcsf, strtoimax(value, &end, BASE));
    }
}

/**
 * @brief Reads a csv list of ints into an array.
 * @param str String list of values.
 * @param val An integer array (set by this function).
 */
static void config_get_ints(char *str, int *val)
{
    int num = 0;
    char *end;
    char *saveptr;
    const char *ptok = strtok_r(str, ARRAY_DELIM, &saveptr);
    while(ptok != NULL) {
        val[num] = strtoimax(ptok, &end, BASE);
        ptok = strtok_r(NULL, ARRAY_DELIM, &saveptr);
        num++;
    }
}

/**
 * @brief Removes tabs/spaces/lf/cr
 * @param s The line to trim.
 */
static void config_trim(char *s) {
    const char *d = s;
    do {
        while(*d == ' ' || *d == '\t' || *d == '\n' || *d == '\r') {
            d++;
        }
    } while((*s++ = *d++));
}

/**
 * @brief Adds a parameter to the list.
 * @param xcsf The XCSF data structure.
 * @param param The parameter to add.
 */
static void config_newnvpair(XCSF *xcsf, const char *param) {
    // get length of name
    size_t namelen = 0;
    _Bool err = true;
    for(namelen = 0; namelen < strnlen(param, MAXLEN); namelen++) {
        if(param[namelen] == '=') {
            err = false;
            break;
        }
    }
    if(err) {
        return; // no '=' found
    }
    // get name
    char *name = malloc(namelen+1);
    for(size_t i = 0; i < namelen; i++) {
        name[i] = param[i];
    }
    name[namelen] = '\0';
    // get value
    size_t valuelen = strnlen(param,MAXLEN)-namelen; // length of value
    char *value = malloc(valuelen+1);
    for(size_t i = 0; i < valuelen; i++) {
        value[i] = param[namelen+1+i];
    }
    value[valuelen] = '\0';
    // add
    config_add_param(xcsf, name, value);
    // clean up
    free(name);
    free(value);
}

/**
 * @brief Parses a line of the config file and adds to the list.
 * @param xcsf The XCSF data structure.
 * @param configline A single line of the configuration file.
 */
static void config_process(XCSF *xcsf, const char *configline) {
    // ignore empty lines
    if(strnlen(configline, MAXLEN) == 0) {
        return;
    }
    // lines starting with # are comments
    if(configline[0] == '#') {
        return; 
    }
    // remove anything after #
    char *ptr = strchr(configline, '#');
    if(ptr != NULL) {
        *ptr = '\0';
    }
    config_newnvpair(xcsf, configline);
}
