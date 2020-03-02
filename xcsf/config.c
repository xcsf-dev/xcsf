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
#include "gp.h"
#include "config.h"
#include "loss.h"

#ifdef PARALLEL
#include <omp.h>
#endif

#define ARRAY_DELIM "," //!< Delimeter for config arrays
#define MAXLEN 127 //!< Maximum config file line length to read
#define BASE 10 //!< Decimal numbers

// reading parameters from configuration files
static void config_get_ints(char *str, int *val);
static void config_newnvpair(XCSF *xcsf, const char *param);
static void config_process(XCSF *xcsf, const char *configline);
static void config_read(XCSF *xcsf, const char *filename);
static void config_trim(char *s);
static void params_cl_action(XCSF *xcsf, const char *name, const char *value);
static void params_cl_condition(XCSF *xcsf, const char *name, char *value);
static void params_cl_general(XCSF *xcsf, const char *name, const char *value);
static void params_cl_prediction(XCSF *xcsf, const char *name, char *value);
static void params_ea(XCSF *xcsf, const char *name, const char *value);
static void params_general(XCSF *xcsf, const char *name, const char *value);
static void params_multistep(XCSF *xcsf, const char *name, const char *value);
static void params_subsumption(XCSF *xcsf, const char *name, const char *value);
// initialising preset parameters
static void config_defaults(XCSF *xcsf);
static void defaults_cl_action(XCSF *xcsf);
static void defaults_cl_condition(XCSF *xcsf);
static void defaults_cl_general(XCSF *xcsf);
static void defaults_cl_prediction(XCSF *xcsf);
static void defaults_ea(XCSF *xcsf);
static void defaults_general(XCSF *xcsf);
static void defaults_multistep(XCSF *xcsf);
static void defaults_subsumption(XCSF *xcsf);
// printing parameters
static void print_params_cl_action(const XCSF *xcsf);
static void print_params_cl_condition(const XCSF *xcsf);
static void print_params_cl_general(const XCSF *xcsf);
static void print_params_cl_prediction(const XCSF *xcsf);
static void print_params_ea(const XCSF *xcsf);
static void print_params_general(const XCSF *xcsf);
static void print_params_multistep(const XCSF *xcsf);
static void print_params_subsumption(const XCSF *xcsf);

/**
 * @brief Initialises global constants and reads the specified configuration file.
 * @param xcsf The XCSF data structure.
 * @param filename The name of the config file to read.
 */
void config_init(XCSF *xcsf, const char *filename)
{
    // initialise parameters
    config_defaults(xcsf);
    config_read(xcsf, filename);
    // initialise (shared) tree-GP constants
    tree_init_cons(xcsf);
    // initialise loss/error function
    loss_set_func(xcsf);
#ifdef PARALLEL
    omp_set_num_threads(xcsf->OMP_NUM_THREADS);
#endif
}

/**
 * @brief Frees all global constants.
 * @param xcsf The XCSF data structure.
 */
void config_free(const XCSF *xcsf)
{
    tree_free_cons(xcsf);
}

/**
 * @brief Sets specified parameter.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void param_add(XCSF *xcsf, const char *name, char *value)
{
    params_general(xcsf, name, value);
    params_multistep(xcsf, name, value);
    params_subsumption(xcsf, name, value);
    params_ea(xcsf, name, value);
    params_cl_general(xcsf, name, value);
    params_cl_condition(xcsf, name, value);
    params_cl_prediction(xcsf, name, value);
    params_cl_action(xcsf, name, value);
}

/**
 * @brief Sets general XCSF parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_general(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "OMP_NUM_THREADS", 16) == 0) {
        xcsf->OMP_NUM_THREADS = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "POP_SIZE", 9) == 0) {
        xcsf->POP_SIZE = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "MAX_TRIALS", 10) == 0) {
        xcsf->MAX_TRIALS = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "POP_INIT", 9) == 0) {
        xcsf->POP_INIT = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->POP_INIT = true;
        }
    }
    else if(strncmp(name, "PERF_TRIALS", 12) == 0) {
        xcsf->PERF_TRIALS = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "LOSS_FUNC", 10) == 0) {
        xcsf->LOSS_FUNC = strtoimax(value, &end, BASE);
    }
}

/**
 * @brief Sets multistep experiment parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_multistep(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "TELETRANSPORTATION", 19) == 0) {
        xcsf->TELETRANSPORTATION = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "GAMMA", 6) == 0) {
        xcsf->GAMMA = atof(value);
    }
    else if(strncmp(name, "P_EXPLORE", 10) == 0) {
        xcsf->P_EXPLORE = atof(value);
    }
}

/**
 * @brief Sets subsumption parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_subsumption(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "EA_SUBSUMPTION", 15) == 0) {
        xcsf->EA_SUBSUMPTION = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->EA_SUBSUMPTION = true;
        }
    }
    else if(strncmp(name, "SET_SUBSUMPTION", 16) == 0) {
        xcsf->SET_SUBSUMPTION = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->SET_SUBSUMPTION = true;
        }
    }
    else if(strncmp(name, "THETA_SUB", 10) == 0) {
        xcsf->THETA_SUB = strtoimax(value, &end, BASE);
    }
}

/**
 * @brief Sets evolutionary algorithm parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_ea(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "EA_SELECT_TYPE", 15) == 0) {
        xcsf->EA_SELECT_TYPE = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "EA_SELECT_SIZE", 15) == 0) {
        xcsf->EA_SELECT_SIZE = atof(value);
    }
    else if(strncmp(name, "THETA_EA", 9) == 0) {
        xcsf->THETA_EA = atof(value);
    }
    else if(strncmp(name, "LAMBDA", 7) == 0) {
        xcsf->LAMBDA = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "P_CROSSOVER", 12) == 0) {
        xcsf->P_CROSSOVER = atof(value);
    }
    else if(strncmp(name, "SAM_TYPE", 9) == 0) {
        xcsf->SAM_TYPE = strtoimax(value, &end, BASE);
    }
}

/**
 * @brief Sets general classifier parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_cl_general(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "ALPHA", 6) == 0) {
        xcsf->ALPHA = atof(value);
    }
    else if(strncmp(name, "BETA", 5) == 0) {
        xcsf->BETA = atof(value);
    }
    else if(strncmp(name, "DELTA", 6) == 0) {
        xcsf->DELTA = atof(value);
    }
    else if(strncmp(name, "NU", 3) == 0) {
        xcsf->NU = atof(value);
    }
    else if(strncmp(name, "THETA_DEL", 10) == 0) {
        xcsf->THETA_DEL = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "INIT_FITNESS", 13) == 0) {
        xcsf->INIT_FITNESS = atof(value);
    }
    else if(strncmp(name, "INIT_ERROR", 11) == 0) {
        xcsf->INIT_ERROR = atof(value);
    }
    else if(strncmp(name, "ERR_REDUC", 10) == 0) {
        xcsf->ERR_REDUC = atof(value);
    }
    else if(strncmp(name, "FIT_REDUC", 10) == 0) {
        xcsf->FIT_REDUC = atof(value);
    }
    else if(strncmp(name, "EPS_0", 6) == 0) {
        xcsf->EPS_0 = atof(value);
    }
    else if(strncmp(name, "M_PROBATION", 12) == 0) {
        xcsf->M_PROBATION = strtoimax(value, &end, BASE);
    }
}

/**
 * @brief Sets classifier condition parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_cl_condition(XCSF *xcsf, const char *name, char *value)
{
    char *end;
    if(strncmp(name, "COND_MIN", 9) == 0) {
        xcsf->COND_MIN = atof(value);
    }
    else if(strncmp(name, "COND_MAX", 9) == 0) {
        xcsf->COND_MAX = atof(value);
    }
    else if(strncmp(name, "COND_TYPE", 10) == 0) {
        xcsf->COND_TYPE = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "COND_SMIN", 10) == 0) {
        xcsf->COND_SMIN = atof(value);
    }
    else if(strncmp(name, "COND_BITS", 10) == 0) {
        xcsf->COND_BITS = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "COND_ETA", 9) == 0) {
        xcsf->COND_ETA = atof(value);
    }
    else if(strncmp(name, "GP_NUM_CONS", 12) == 0) {
        xcsf->GP_NUM_CONS = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "GP_INIT_DEPTH", 14) == 0) {
        xcsf->GP_INIT_DEPTH = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "DGP_NUM_NODES", 14) == 0) {
        xcsf->DGP_NUM_NODES = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "RESET_STATES", 13) == 0) {
        xcsf->RESET_STATES = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->RESET_STATES = true;
        }
    }
    else if(strncmp(name, "MAX_K", 6) == 0) {
        xcsf->MAX_K = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "MAX_T", 6) == 0) {
        xcsf->MAX_T = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "MAX_NEURON_MOD", 15) == 0) {
        xcsf->MAX_NEURON_MOD = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "COND_EVOLVE_WEIGHTS", 20) == 0) {
        xcsf->COND_EVOLVE_WEIGHTS = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->COND_EVOLVE_WEIGHTS = true;
        }
    }
    else if(strncmp(name, "COND_EVOLVE_NEURONS", 20) == 0) {
        xcsf->COND_EVOLVE_NEURONS = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->COND_EVOLVE_NEURONS = true;
        }
    }
    else if(strncmp(name, "COND_EVOLVE_FUNCTIONS", 22) == 0) {
        xcsf->COND_EVOLVE_FUNCTIONS = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->COND_EVOLVE_FUNCTIONS = true;
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
        xcsf->COND_OUTPUT_ACTIVATION = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "COND_HIDDEN_ACTIVATION", 23) == 0) {
        xcsf->COND_HIDDEN_ACTIVATION = strtoimax(value, &end, BASE);
    }
}

/**
 * @brief Sets classifier prediction parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_cl_prediction(XCSF *xcsf, const char *name, char *value)
{
    char *end;
    if(strncmp(name, "PRED_TYPE", 10) == 0) {
        xcsf->PRED_TYPE = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "PRED_ETA", 9) == 0) {
        xcsf->PRED_ETA = atof(value);
    }
    else if(strncmp(name, "PRED_RESET", 11) == 0) {
        xcsf->PRED_RESET = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->PRED_RESET = true;
        }
    }
    else if(strncmp(name, "PRED_X0", 8) == 0) {
        xcsf->PRED_X0 = atof(value);
    }
    else if(strncmp(name, "PRED_RLS_SCALE_FACTOR", 22) == 0) {
        xcsf->PRED_RLS_SCALE_FACTOR = atof(value);
    }
    else if(strncmp(name, "PRED_RLS_LAMBDA", 16) == 0) {
        xcsf->PRED_RLS_LAMBDA = atof(value);
    }
    else if(strncmp(name, "PRED_EVOLVE_WEIGHTS", 20) == 0) {
        xcsf->PRED_EVOLVE_WEIGHTS = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->PRED_EVOLVE_WEIGHTS = true;
        }
    }
    else if(strncmp(name, "PRED_EVOLVE_NEURONS", 20) == 0) {
        xcsf->PRED_EVOLVE_NEURONS = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->PRED_EVOLVE_NEURONS = true;
        }
    }
    else if(strncmp(name, "PRED_EVOLVE_FUNCTIONS", 22) == 0) {
        xcsf->PRED_EVOLVE_FUNCTIONS = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->PRED_EVOLVE_FUNCTIONS = true;
        }
    }
    else if(strncmp(name, "PRED_EVOLVE_ETA", 16) == 0) {
        xcsf->PRED_EVOLVE_ETA = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->PRED_EVOLVE_ETA = true;
        }
    }
    else if(strncmp(name, "PRED_SGD_WEIGHTS", 17) == 0) {
        xcsf->PRED_SGD_WEIGHTS = false;
        if(strncmp(value, "true", 5) == 0) {
            xcsf->PRED_SGD_WEIGHTS = true;
        }
    }
    else if(strncmp(name, "PRED_MOMENTUM", 14) == 0) {
        xcsf->PRED_MOMENTUM = atof(value);
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
        xcsf->PRED_OUTPUT_ACTIVATION = strtoimax(value, &end, BASE);
    }
    else if(strncmp(name, "PRED_HIDDEN_ACTIVATION", 23) == 0) {
        xcsf->PRED_HIDDEN_ACTIVATION = strtoimax(value, &end, BASE);
    }
}

/**
 * @brief Sets classifier action parameters.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void params_cl_action(XCSF *xcsf, const char *name, const char *value)
{
    char *end;
    if(strncmp(name, "ACT_TYPE", 9) == 0) {
        xcsf->ACT_TYPE = strtoimax(value, &end, BASE);
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
    param_add(xcsf, name, value);
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

/**
 * @brief Reads the specified configuration file.
 * @param xcsf The XCSF data structure.
 * @param filename The name of the configuration file.
 */
static void config_read(XCSF *xcsf, const char *filename) {
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
 * @brief Initialises default XCSF parameters.
 */
static void config_defaults(XCSF *xcsf)
{
    defaults_cl_action(xcsf);
    defaults_cl_condition(xcsf);
    defaults_cl_general(xcsf);
    defaults_cl_prediction(xcsf);
    defaults_ea(xcsf);
    defaults_general(xcsf);
    defaults_multistep(xcsf);
    defaults_subsumption(xcsf);
}

/**
 * @brief Initialises default XCSF general parameters.
 */
static void defaults_general(XCSF *xcsf)
{
    xcsf->OMP_NUM_THREADS = 8;
    xcsf->POP_SIZE = 2000;
    xcsf->MAX_TRIALS = 100000;
    xcsf->POP_INIT = true;
    xcsf->PERF_TRIALS = 1000;
    xcsf->LOSS_FUNC = 0;
}

/**
 * @brief Initialises default multistep parameters.
 */
static void defaults_multistep(XCSF *xcsf)
{
    xcsf->TELETRANSPORTATION = 50;
    xcsf->GAMMA = 0.95;
    xcsf->P_EXPLORE = 0.9;
}

/**
 * @brief Initialises default general classifier parameters.
 */
static void defaults_cl_general(XCSF *xcsf)
{
    xcsf->EPS_0 = 0.01;
    xcsf->ALPHA = 0.1;
    xcsf->NU = 5;
    xcsf->BETA = 0.1;
    xcsf->DELTA = 0.1;
    xcsf->THETA_DEL = 20;
    xcsf->INIT_FITNESS = 0.01;
    xcsf->INIT_ERROR = 0;
    xcsf->ERR_REDUC = 1;
    xcsf->FIT_REDUC = 0.1;
    xcsf->M_PROBATION = 10000;
}

/**
 * @brief Initialises default subsumption parameters.
 */
static void defaults_subsumption(XCSF *xcsf)
{
    xcsf->EA_SUBSUMPTION = false;
    xcsf->SET_SUBSUMPTION = false;
    xcsf->THETA_SUB = 1000;
}

/**
 * @brief Initialises default evolutionary algorithm parameters.
 */
static void defaults_ea(XCSF *xcsf)
{
    xcsf->EA_SELECT_TYPE = 0;
    xcsf->EA_SELECT_SIZE = 0.4;
    xcsf->THETA_EA = 50;
    xcsf->LAMBDA = 2;
    xcsf->P_CROSSOVER = 0.8;
    xcsf->SAM_TYPE = 0;
}

/**
 * @brief Initialises default classifier condition parameters.
 */
static void defaults_cl_condition(XCSF *xcsf)
{
    xcsf->COND_ETA = 0;
    xcsf->COND_TYPE = 1;
    xcsf->COND_MIN = 0;
    xcsf->COND_MAX = 1;
    xcsf->COND_SMIN = 0.1;
    xcsf->COND_BITS = 1;
    xcsf->GP_NUM_CONS = 100;
    xcsf->GP_INIT_DEPTH = 5;
    xcsf->DGP_NUM_NODES = 20;
    xcsf->RESET_STATES = false;
    xcsf->MAX_K = 2;
    xcsf->MAX_T = 10;
    xcsf->MAX_NEURON_MOD = 1;
    xcsf->COND_EVOLVE_WEIGHTS = true;
    xcsf->COND_EVOLVE_NEURONS = true;
    xcsf->COND_EVOLVE_FUNCTIONS = false;
    memset(xcsf->COND_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
    memset(xcsf->COND_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
    xcsf->COND_NUM_NEURONS[0] = 1;
    xcsf->COND_MAX_NEURONS[0] = 10;
    xcsf->COND_OUTPUT_ACTIVATION = 0;
    xcsf->COND_HIDDEN_ACTIVATION = 0;
}

/**
 * @brief Initialises default classifier prediction parameters.
 */
static void defaults_cl_prediction(XCSF *xcsf)
{
    xcsf->PRED_TYPE = 1;
    xcsf->PRED_EVOLVE_ETA = true;
    xcsf->PRED_ETA = 0.1;
    xcsf->PRED_RESET = false;
    xcsf->PRED_X0 = 1;
    xcsf->PRED_RLS_SCALE_FACTOR = 1000;
    xcsf->PRED_RLS_LAMBDA = 1;
    xcsf->PRED_EVOLVE_WEIGHTS = true;
    xcsf->PRED_EVOLVE_NEURONS = true;
    xcsf->PRED_EVOLVE_FUNCTIONS = false;
    xcsf->PRED_SGD_WEIGHTS = true;
    xcsf->PRED_MOMENTUM = 0.9;
    memset(xcsf->PRED_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
    memset(xcsf->PRED_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
    xcsf->PRED_NUM_NEURONS[0] = 1;
    xcsf->PRED_MAX_NEURONS[0] = 10;
    xcsf->PRED_OUTPUT_ACTIVATION = 0;
    xcsf->PRED_HIDDEN_ACTIVATION = 0;
}

/**
 * @brief Initialises default classifier action parameters.
 */
static void defaults_cl_action(XCSF *xcsf)
{
    xcsf->ACT_TYPE = 0;
}

/**
 * @brief Prints all XCSF parameters.
 * @param xcsf The XCSF data structure.
 */
void config_print(const XCSF *xcsf)
{
    print_params_general(xcsf);
    print_params_multistep(xcsf);
    print_params_ea(xcsf);
    print_params_subsumption(xcsf);
    print_params_cl_general(xcsf);
    print_params_cl_condition(xcsf);
    print_params_cl_prediction(xcsf);
    print_params_cl_action(xcsf);
    printf("\n");
}

/**
 * @brief Prints XCSF general parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_general(const XCSF *xcsf)
{
    printf("OMP_NUM_THREADS=%d", xcsf->OMP_NUM_THREADS);
    printf(", POP_SIZE=%d", xcsf->POP_SIZE);
    printf(", MAX_TRIALS=%d", xcsf->MAX_TRIALS);
    printf(", POP_INIT=");
    xcsf->POP_INIT == true ? printf("true") : printf("false");
    printf(", PERF_TRIALS=%d", xcsf->PERF_TRIALS);
    printf(", LOSS_FUNC=%d", xcsf->LOSS_FUNC);
}

/**
 * @brief Prints XCSF multistep parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_multistep(const XCSF *xcsf)
{
    printf(", TELETRANSPORTATION=%d", xcsf->TELETRANSPORTATION);
    printf(", GAMMA=%f", xcsf->GAMMA);
    printf(", P_EXPLORE=%f", xcsf->P_EXPLORE);
}

/**
 * @brief Prints XCSF general classifier parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_cl_general(const XCSF *xcsf)
{
    printf(", EPS_0=%f", xcsf->EPS_0);
    printf(", ALPHA=%f", xcsf->ALPHA);
    printf(", NU=%f", xcsf->NU);
    printf(", BETA=%f", xcsf->BETA);
    printf(", DELTA=%f", xcsf->DELTA);
    printf(", THETA_DEL=%d", xcsf->THETA_DEL);
    printf(", INIT_FITNESS=%f", xcsf->INIT_FITNESS);
    printf(", INIT_ERROR=%f", xcsf->INIT_ERROR);
    printf(", ERR_REDUC=%f", xcsf->ERR_REDUC);
    printf(", FIT_REDUC=%f", xcsf->FIT_REDUC);
    printf(", M_PROBATION=%d", xcsf->M_PROBATION);
}

/**
 * @brief Prints XCSF subsumption parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_subsumption(const XCSF *xcsf)
{
    printf(", EA_SUBSUMPTION=");
    xcsf->EA_SUBSUMPTION == true ? printf("true") : printf("false");
    printf(", SET_SUBSUMPTION=");
    xcsf->SET_SUBSUMPTION == true ? printf("true") : printf("false");
    printf(", THETA_SUB=%d", xcsf->THETA_SUB);
}

/**
 * @brief Prints XCSF evolutionary algorithm parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_ea(const XCSF *xcsf)
{
    printf(", EA_SELECT_TYPE=%d", xcsf->EA_SELECT_TYPE);
    printf(", EA_SELECT_SIZE=%f", xcsf->EA_SELECT_SIZE);
    printf(", THETA_EA=%f", xcsf->THETA_EA);
    printf(", LAMBDA=%d", xcsf->LAMBDA);
    printf(", P_CROSSOVER=%f", xcsf->P_CROSSOVER);
    printf(", SAM_TYPE=%d", xcsf->SAM_TYPE);
}

/**
 * @brief Prints XCSF condtion parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_cl_condition(const XCSF *xcsf)
{
    printf(", COND_ETA=%f", xcsf->COND_ETA);
    printf(", COND_TYPE=%d", xcsf->COND_TYPE);
    printf(", COND_MIN=%f", xcsf->COND_MIN);
    printf(", COND_MAX=%f", xcsf->COND_MAX);
    printf(", COND_SMIN=%f", xcsf->COND_SMIN);
    printf(", COND_BITS=%d", xcsf->COND_BITS);
    printf(", GP_NUM_CONS=%d", xcsf->GP_NUM_CONS);
    printf(", GP_INIT_DEPTH=%d", xcsf->GP_INIT_DEPTH);
    printf(", DGP_NUM_NODES=%d", xcsf->DGP_NUM_NODES);
    printf(", RESET_STATES=");
    xcsf->RESET_STATES == true ? printf("true") : printf("false");
    printf(", MAX_K=%d", xcsf->MAX_K);
    printf(", MAX_T=%d", xcsf->MAX_T);
    printf(", MAX_NEURON_MOD=%d", xcsf->MAX_NEURON_MOD);
    printf(", COND_EVOLVE_WEIGHTS=");
    xcsf->COND_EVOLVE_WEIGHTS == true ? printf("true") : printf("false");
    printf(", COND_EVOLVE_NEURONS=");
    xcsf->COND_EVOLVE_NEURONS == true ? printf("true") : printf("false");
    printf(", COND_EVOLVE_FUNCTIONS=");
    xcsf->COND_EVOLVE_FUNCTIONS == true ? printf("true") : printf("false");
    printf(", COND_NUM_NEURONS=[");
    for(int i = 0;  i < MAX_LAYERS && xcsf->COND_NUM_NEURONS[i] > 0; i++) {
        printf("%d;", xcsf->COND_NUM_NEURONS[i]);
    }
    printf("]");
    printf(", COND_MAX_NEURONS=[");
    for(int i = 0;  i < MAX_LAYERS && xcsf->COND_MAX_NEURONS[i] > 0; i++) {
        printf("%d;", xcsf->COND_MAX_NEURONS[i]);
    }
    printf("]");
    printf(", COND_OUTPUT_ACTIVATION=%d", xcsf->COND_OUTPUT_ACTIVATION);
    printf(", COND_HIDDEN_ACTIVATION=%d", xcsf->COND_HIDDEN_ACTIVATION);
}

/**
 * @brief Prints XCSF prediction parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_cl_prediction(const XCSF *xcsf)
{
    printf(", PRED_TYPE=%d", xcsf->PRED_TYPE);
    printf(", PRED_EVOLVE_ETA=");
    xcsf->PRED_EVOLVE_ETA == true ? printf("true") : printf("false");
    printf(", PRED_ETA=%f", xcsf->PRED_ETA);
    printf(", PRED_RESET=");
    xcsf->PRED_RESET == true ? printf("true") : printf("false");
    printf(", PRED_X0=%f", xcsf->PRED_X0);
    printf(", PRED_RLS_SCALE_FACTOR=%f", xcsf->PRED_RLS_SCALE_FACTOR);
    printf(", PRED_RLS_SCALE_LAMBDA=%f", xcsf->PRED_RLS_LAMBDA);
    printf(", PRED_EVOLVE_WEIGHTS=");
    xcsf->PRED_EVOLVE_WEIGHTS == true ? printf("true") : printf("false");
    printf(", PRED_EVOLVE_NEURONS=");
    xcsf->PRED_EVOLVE_NEURONS == true ? printf("true") : printf("false");
    printf(", PRED_EVOLVE_FUNCTIONS=");
    xcsf->PRED_EVOLVE_FUNCTIONS == true ? printf("true") : printf("false");
    printf(", PRED_SGD_WEIGHTS=");
    xcsf->PRED_SGD_WEIGHTS == true ? printf("true") : printf("false");
    printf(", PRED_MOMENTUM=%f", xcsf->PRED_MOMENTUM);
    printf(", PRED_NUM_NEURONS=[");
    for(int i = 0;  i < MAX_LAYERS && xcsf->PRED_NUM_NEURONS[i] > 0; i++) {
        printf("%d;", xcsf->PRED_NUM_NEURONS[i]);
    }
    printf("]");
    printf(", PRED_MAX_NEURONS=[");
    for(int i = 0;  i < MAX_LAYERS && xcsf->PRED_MAX_NEURONS[i] > 0; i++) {
        printf("%d;", xcsf->PRED_MAX_NEURONS[i]);
    }
    printf("]");
    printf(", PRED_OUTPUT_ACTIVATION=%d", xcsf->PRED_OUTPUT_ACTIVATION);
    printf(", PRED_HIDDEN_ACTIVATION=%d", xcsf->PRED_HIDDEN_ACTIVATION);
}

/**
 * @brief Prints XCSF action parameters.
 * @param xcsf The XCSF data structure.
 */
static void print_params_cl_action(const XCSF *xcsf)
{
    printf(", ACT_TYPE=%d", xcsf->ACT_TYPE);
}
