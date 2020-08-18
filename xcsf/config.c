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

#include "config.h"
#include "param.h"

#define ARRAY_DELIM (",") //!< Delimeter for config arrays
#define MAXLEN (127) //!< Maximum config file line length to read
#define BASE (10) //!< Decimal numbers

/**
 * @brief Reads a csv list of ints into an array.
 * @param str String list of values.
 * @param val An integer array (set by this function).
 */
static void
config_get_ints(char *str, int *val)
{
    int num = 0;
    char *end = NULL;
    char *saveptr = NULL;
    const char *ptok = strtok_r(str, ARRAY_DELIM, &saveptr);
    while (ptok != NULL) {
        val[num] = (int) strtoimax(ptok, &end, BASE);
        ptok = strtok_r(NULL, ARRAY_DELIM, &saveptr);
        ++num;
    }
}

/**
 * @brief Sets general XCSF parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_general(struct XCSF *xcsf, const char *n, const char *v, int i, double f)
{
    (void) v;
    (void) f;
    if (strncmp(n, "OMP_NUM_THREADS\0", 16) == 0) {
        param_set_omp_num_threads(xcsf, i);
    } else if (strncmp(n, "POP_SIZE\0", 9) == 0) {
        param_set_pop_size(xcsf, i);
    } else if (strncmp(n, "MAX_TRIALS\0", 10) == 0) {
        param_set_max_trials(xcsf, i);
    } else if (strncmp(n, "POP_INIT\0", 9) == 0) {
        param_set_pop_init(xcsf, i);
    } else if (strncmp(n, "PERF_TRIALS\0", 12) == 0) {
        param_set_perf_trials(xcsf, i);
    } else if (strncmp(n, "LOSS_FUNC\0", 10) == 0) {
        param_set_loss_func(xcsf, i);
    }
}

/**
 * @brief Sets multistep experiment parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_multi(struct XCSF *xcsf, const char *n, const char *v, int i, double f)
{
    (void) v;
    if (strncmp(n, "TELETRANSPORTATION\0", 19) == 0) {
        param_set_teletransportation(xcsf, i);
    } else if (strncmp(n, "GAMMA\0", 6) == 0) {
        param_set_gamma(xcsf, f);
    } else if (strncmp(n, "P_EXPLORE\0", 10) == 0) {
        param_set_p_explore(xcsf, f);
    }
}

/**
 * @brief Sets subsumption parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_subsump(struct XCSF *xcsf, const char *n, const char *v, int i, double f)
{
    (void) v;
    (void) f;
    if (strncmp(n, "EA_SUBSUMPTION\0", 15) == 0) {
        param_set_ea_subsumption(xcsf, i);
    } else if (strncmp(n, "SET_SUBSUMPTION\0", 16) == 0) {
        param_set_set_subsumption(xcsf, i);
    } else if (strncmp(n, "THETA_SUB\0", 10) == 0) {
        param_set_theta_sub(xcsf, i);
    }
}

/**
 * @brief Sets evolutionary algorithm parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_ea(struct XCSF *xcsf, const char *n, const char *v, int i, double f)
{
    (void) v;
    if (strncmp(n, "EA_SELECT_TYPE\0", 15) == 0) {
        param_set_ea_select_type(xcsf, i);
    } else if (strncmp(n, "EA_SELECT_SIZE\0", 15) == 0) {
        param_set_ea_select_size(xcsf, f);
    } else if (strncmp(n, "THETA_EA\0", 9) == 0) {
        param_set_theta_ea(xcsf, f);
    } else if (strncmp(n, "LAMBDA\0", 7) == 0) {
        param_set_lambda(xcsf, i);
    } else if (strncmp(n, "P_CROSSOVER\0", 12) == 0) {
        param_set_p_crossover(xcsf, f);
    }
}

/**
 * @brief Sets general classifier parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_cl_gen(struct XCSF *xcsf, const char *n, const char *v, int i, double f)
{
    (void) v;
    if (strncmp(n, "ALPHA\0", 6) == 0) {
        param_set_alpha(xcsf, f);
    } else if (strncmp(n, "BETA\0", 5) == 0) {
        param_set_beta(xcsf, f);
    } else if (strncmp(n, "DELTA\0", 6) == 0) {
        param_set_delta(xcsf, f);
    } else if (strncmp(n, "NU\0", 3) == 0) {
        param_set_nu(xcsf, f);
    } else if (strncmp(n, "THETA_DEL\0", 10) == 0) {
        param_set_theta_del(xcsf, i);
    } else if (strncmp(n, "INIT_FITNESS\0", 13) == 0) {
        param_set_init_fitness(xcsf, f);
    } else if (strncmp(n, "INIT_ERROR\0", 11) == 0) {
        param_set_init_error(xcsf, f);
    } else if (strncmp(n, "ERR_REDUC\0", 10) == 0) {
        param_set_err_reduc(xcsf, f);
    } else if (strncmp(n, "FIT_REDUC\0", 10) == 0) {
        param_set_fit_reduc(xcsf, f);
    } else if (strncmp(n, "EPS_0\0", 6) == 0) {
        param_set_eps_0(xcsf, f);
    } else if (strncmp(n, "M_PROBATION\0", 12) == 0) {
        param_set_m_probation(xcsf, i);
    } else if (strncmp(n, "STATEFUL\0", 9) == 0) {
        param_set_stateful(xcsf, i);
    }
}

/**
 * @brief Sets classifier condition parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_cl_cond(struct XCSF *xcsf, const char *n, char *v, int i, double f)
{
    if (strncmp(n, "COND_TYPE\0", 10) == 0) {
        param_set_cond_type(xcsf, i);
    } else if (strncmp(n, "COND_MIN\0", 9) == 0) {
        param_set_cond_min(xcsf, f);
    } else if (strncmp(n, "COND_MAX\0", 9) == 0) {
        param_set_cond_max(xcsf, f);
    } else if (strncmp(n, "COND_SMIN\0", 10) == 0) {
        param_set_cond_smin(xcsf, f);
    } else if (strncmp(n, "COND_BITS\0", 10) == 0) {
        param_set_cond_bits(xcsf, i);
    } else if (strncmp(n, "COND_ETA\0", 9) == 0) {
        param_set_cond_eta(xcsf, f);
    } else if (strncmp(n, "GP_NUM_CONS\0", 12) == 0) {
        param_set_gp_num_cons(xcsf, i);
    } else if (strncmp(n, "GP_INIT_DEPTH\0", 14) == 0) {
        param_set_gp_init_depth(xcsf, i);
    } else if (strncmp(n, "MAX_K\0", 6) == 0) {
        param_set_max_k(xcsf, i);
    } else if (strncmp(n, "MAX_T\0", 6) == 0) {
        param_set_max_t(xcsf, i);
    } else if (strncmp(n, "MAX_NEURON_GROW\0", 16) == 0) {
        param_set_max_neuron_grow(xcsf, i);
    } else if (strncmp(n, "COND_EVOLVE_WEIGHTS\0", 20) == 0) {
        param_set_cond_evolve_weights(xcsf, i);
    } else if (strncmp(n, "COND_EVOLVE_NEURONS\0", 20) == 0) {
        param_set_cond_evolve_neurons(xcsf, i);
    } else if (strncmp(n, "COND_EVOLVE_FUNCTIONS\0", 22) == 0) {
        param_set_cond_evolve_functions(xcsf, i);
    } else if (strncmp(n, "COND_NUM_NEURONS\0", 17) == 0) {
        memset(xcsf->COND_NUM_NEURONS, 0, sizeof(int) * MAX_LAYERS);
        config_get_ints(v, xcsf->COND_NUM_NEURONS);
    } else if (strncmp(n, "COND_MAX_NEURONS\0", 17) == 0) {
        memset(xcsf->COND_MAX_NEURONS, 0, sizeof(int) * MAX_LAYERS);
        config_get_ints(v, xcsf->COND_MAX_NEURONS);
    } else if (strncmp(n, "COND_OUTPUT_ACTIVATION\0", 23) == 0) {
        param_set_cond_output_activation(xcsf, i);
    } else if (strncmp(n, "COND_HIDDEN_ACTIVATION\0", 23) == 0) {
        param_set_cond_hidden_activation(xcsf, i);
    }
}

/**
 * @brief Sets classifier prediction parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_cl_pred(struct XCSF *xcsf, const char *n, char *v, int i, double f)
{
    if (strncmp(n, "PRED_TYPE\0", 10) == 0) {
        param_set_pred_type(xcsf, i);
    } else if (strncmp(n, "PRED_ETA\0", 9) == 0) {
        param_set_pred_eta(xcsf, f);
    } else if (strncmp(n, "PRED_RESET\0", 11) == 0) {
        param_set_pred_reset(xcsf, i);
    } else if (strncmp(n, "PRED_X0\0", 8) == 0) {
        param_set_pred_x0(xcsf, f);
    } else if (strncmp(n, "PRED_RLS_SCALE_FACTOR\0", 22) == 0) {
        param_set_pred_rls_scale_factor(xcsf, f);
    } else if (strncmp(n, "PRED_RLS_LAMBDA\0", 16) == 0) {
        param_set_pred_rls_lambda(xcsf, f);
    } else if (strncmp(n, "PRED_EVOLVE_WEIGHTS\0", 20) == 0) {
        param_set_pred_evolve_weights(xcsf, i);
    } else if (strncmp(n, "PRED_EVOLVE_NEURONS\0", 20) == 0) {
        param_set_pred_evolve_neurons(xcsf, i);
    } else if (strncmp(n, "PRED_EVOLVE_FUNCTIONS\0", 22) == 0) {
        param_set_pred_evolve_functions(xcsf, i);
    } else if (strncmp(n, "PRED_EVOLVE_ETA\0", 16) == 0) {
        param_set_pred_evolve_eta(xcsf, i);
    } else if (strncmp(n, "PRED_SGD_WEIGHTS\0", 17) == 0) {
        param_set_pred_sgd_weights(xcsf, i);
    } else if (strncmp(n, "PRED_MOMENTUM\0", 14) == 0) {
        param_set_pred_momentum(xcsf, f);
    } else if (strncmp(n, "PRED_NUM_NEURONS\0", 17) == 0) {
        memset(xcsf->PRED_NUM_NEURONS, 0, sizeof(int) * MAX_LAYERS);
        config_get_ints(v, xcsf->PRED_NUM_NEURONS);
    } else if (strncmp(n, "PRED_MAX_NEURONS\0", 17) == 0) {
        memset(xcsf->PRED_MAX_NEURONS, 0, sizeof(int) * MAX_LAYERS);
        config_get_ints(v, xcsf->PRED_MAX_NEURONS);
    } else if (strncmp(n, "PRED_OUTPUT_ACTIVATION\0", 23) == 0) {
        param_set_pred_output_activation(xcsf, i);
    } else if (strncmp(n, "PRED_HIDDEN_ACTIVATION\0", 23) == 0) {
        param_set_pred_hidden_activation(xcsf, i);
    }
}

/**
 * @brief Sets classifier action parameters.
 * @param xcsf The XCSF data structure.
 * @param n String representation of the parameter name.
 * @param v String representation of the parameter value.
 * @param i Integer representation of the parameter value.
 * @param f Float representation of the parameter value.
 */
static void
config_cl_act(struct XCSF *xcsf, const char *n, const char *v, int i, double f)
{
    (void) v;
    (void) f;
    if (strncmp(n, "ACT_TYPE\0", 9) == 0) {
        param_set_act_type(xcsf, i);
    }
}

/**
 * @brief Sets specified parameter.
 * @param xcsf The XCSF data structure.
 * @param name Parameter name.
 * @param value Parameter value.
 */
static void
config_add_param(struct XCSF *xcsf, const char *name, char *value)
{
    int i = 0;
    char *endptr = NULL;
    if (strncmp(value, "true\0", 5) == 0) {
        i = 1;
    } else if (strncmp(value, "false\0", 6) == 0) {
        i = 0;
    } else {
        i = (int) strtoimax(value, &endptr, BASE);
    }
    double f = strtod(value, &endptr);
    // add parameter
    config_general(xcsf, name, value, i, f);
    config_multi(xcsf, name, value, i, f);
    config_subsump(xcsf, name, value, i, f);
    config_ea(xcsf, name, value, i, f);
    config_cl_gen(xcsf, name, value, i, f);
    config_cl_cond(xcsf, name, value, i, f);
    config_cl_pred(xcsf, name, value, i, f);
    config_cl_act(xcsf, name, value, i, f);
}

/**
 * @brief Removes tabs/spaces/lf/cr
 * @param s The line to trim.
 */
static void
config_trim(char *s)
{
    const char *d = s;
    do {
        while (*d == ' ' || *d == '\t' || *d == '\n' || *d == '\r') {
            ++d;
        }
    } while ((*s++ = *d++));
}

/**
 * @brief Adds a parameter to the list.
 * @param xcsf The XCSF data structure.
 * @param param The parameter to add.
 */
static void
config_newnvpair(struct XCSF *xcsf, const char *param)
{
    // get length of name
    size_t namelen = 0;
    _Bool err = true;
    for (namelen = 0; namelen < strnlen(param, MAXLEN); ++namelen) {
        if (param[namelen] == '=') {
            err = false;
            break;
        }
    }
    if (err) {
        return; // no '=' found
    }
    // get name
    char *name = malloc(namelen + 1);
    for (size_t i = 0; i < namelen; ++i) {
        name[i] = param[i];
    }
    name[namelen] = '\0';
    // get value
    size_t valuelen = strnlen(param, MAXLEN) - namelen; // length of value
    char *value = malloc(valuelen + 1);
    for (size_t i = 0; i < valuelen; ++i) {
        value[i] = param[namelen + 1 + i];
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
static void
config_process(struct XCSF *xcsf, const char *configline)
{
    // ignore empty lines
    if (strnlen(configline, MAXLEN) == 0) {
        return;
    }
    // lines starting with # are comments
    if (configline[0] == '#') {
        return;
    }
    // remove anything after #
    char *ptr = strchr(configline, '#');
    if (ptr != NULL) {
        *ptr = '\0';
    }
    config_newnvpair(xcsf, configline);
}

/**
 * @brief Reads the specified configuration file.
 * @param xcsf The XCSF data structure.
 * @param filename The name of the configuration file.
 */
void
config_read(struct XCSF *xcsf, const char *filename)
{
    FILE *f = fopen(filename, "rte");
    if (f == NULL) {
        printf("Warning: could not open %s.\n", filename);
        return;
    }
    char buff[MAXLEN];
    while (!feof(f)) {
        if (fgets(buff, MAXLEN - 2, f) == NULL) {
            break;
        }
        config_trim(buff);
        config_process(xcsf, buff);
    }
    fclose(f);
}
