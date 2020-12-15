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
#include "action.h"
#include "condition.h"
#include "dgp.h"
#include "ea.h"
#include "gp.h"
#include "neural_activations.h"
#include "neural_layer.h"
#include "param.h"
#include "prediction.h"

#define MAXLEN (127) //!< Maximum config file line length to read
#define BASE (10) //!< Decimal numbers

struct ArgsLayer *current_layer; //!< Current layer parameters being read

/**
 * @brief Sets general XCSF parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_general(struct XCSF *xcsf, const char *n, const char *v, const int i,
               const double f)
{
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
        param_set_loss_func_string(xcsf, v);
    } else if (strncmp(n, "HUBER_DELTA\0", 12) == 0) {
        param_set_huber_delta(xcsf, f);
    }
}

/**
 * @brief Sets multistep experiment parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_multi(struct XCSF *xcsf, const char *n, const char *v, const int i,
             const double f)
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
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_subsump(struct XCSF *xcsf, const char *n, const char *v, const int i,
               const double f)
{
    (void) v;
    (void) f;
    if (strncmp(n, "SET_SUBSUMPTION\0", 16) == 0) {
        param_set_set_subsumption(xcsf, i);
    } else if (strncmp(n, "THETA_SUB\0", 10) == 0) {
        param_set_theta_sub(xcsf, i);
    }
}

/**
 * @brief Sets evolutionary algorithm parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_ea(struct XCSF *xcsf, const char *n, const char *v, const int i,
          const double f)
{
    if (strncmp(n, "EA_SELECT_TYPE\0", 15) == 0) {
        ea_param_set_type_string(xcsf, v);
    } else if (strncmp(n, "EA_SELECT_SIZE\0", 15) == 0) {
        ea_param_set_select_size(xcsf, f);
    } else if (strncmp(n, "THETA_EA\0", 9) == 0) {
        ea_param_set_theta(xcsf, f);
    } else if (strncmp(n, "LAMBDA\0", 7) == 0) {
        ea_param_set_lambda(xcsf, i);
    } else if (strncmp(n, "P_CROSSOVER\0", 12) == 0) {
        ea_param_set_p_crossover(xcsf, f);
    } else if (strncmp(n, "ERR_REDUC\0", 10) == 0) {
        ea_param_set_err_reduc(xcsf, f);
    } else if (strncmp(n, "FIT_REDUC\0", 10) == 0) {
        ea_param_set_fit_reduc(xcsf, f);
    } else if (strncmp(n, "EA_PRED_RESET\0", 14) == 0) {
        ea_param_set_pred_reset(xcsf, i);
    }
}

/**
 * @brief Sets general classifier parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_gen(struct XCSF *xcsf, const char *n, const char *v, const int i,
              const double f)
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
    } else if (strncmp(n, "E0\0", 3) == 0) {
        param_set_e0(xcsf, f);
    } else if (strncmp(n, "M_PROBATION\0", 12) == 0) {
        param_set_m_probation(xcsf, i);
    } else if (strncmp(n, "STATEFUL\0", 9) == 0) {
        param_set_stateful(xcsf, i);
    } else if (strncmp(n, "COMPACTION\0", 11) == 0) {
        param_set_compaction(xcsf, i);
    }
}

/**
 * @brief Sets neural network layer parameters.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_layer(const char *n, const char *v, const int i, const double f)
{
    if (strncmp(n, "LAYER_ACTIVATION\0", 17) == 0) {
        current_layer->function = neural_activation_as_int(v);
    } else if (strncmp(n, "LAYER_RECURRENT_ACTIVATION\0", 27) == 0) {
        current_layer->recurrent_function = neural_activation_as_int(v);
    } else if (strncmp(n, "LAYER_N_INIT\0", 13) == 0) {
        current_layer->n_init = i;
    } else if (strncmp(n, "LAYER_N_MAX\0", 12) == 0) {
        current_layer->n_max = i;
    } else if (strncmp(n, "LAYER_N_INPUTS\0", 15) == 0) {
        current_layer->n_inputs = i;
    } else if (strncmp(n, "LAYER_MAX_NEURON_GROW\0", 22) == 0) {
        current_layer->max_neuron_grow = i;
    } else if (strncmp(n, "LAYER_EVOLVE_WEIGHTS\0", 21) == 0) {
        current_layer->evolve_weights = i;
    } else if (strncmp(n, "LAYER_EVOLVE_NEURONS\0", 21) == 0) {
        current_layer->evolve_neurons = i;
    } else if (strncmp(n, "LAYER_EVOLVE_FUNCTIONS\0", 23) == 0) {
        current_layer->evolve_functions = i;
    } else if (strncmp(n, "LAYER_EVOLVE_CONNECT\0", 21) == 0) {
        current_layer->evolve_connect = i;
    } else if (strncmp(n, "LAYER_EVOLVE_ETA\0", 17) == 0) {
        current_layer->evolve_eta = i;
    } else if (strncmp(n, "LAYER_SGD_WEIGHTS\0", 18) == 0) {
        current_layer->sgd_weights = i;
    } else if (strncmp(n, "LAYER_ETA\0", 10) == 0) {
        current_layer->eta = f;
    } else if (strncmp(n, "LAYER_ETA_MIN\0", 14) == 0) {
        current_layer->eta_min = f;
    } else if (strncmp(n, "LAYER_MOMENTUM\0", 15) == 0) {
        current_layer->momentum = f;
    } else if (strncmp(n, "LAYER_DECAY\0", 12) == 0) {
        current_layer->decay = f;
    } else if (strncmp(n, "LAYER_SCALE\0", 12) == 0) {
        current_layer->scale = f;
    } else if (strncmp(n, "LAYER_WIDTH\0", 12) == 0) {
        current_layer->width = i;
    } else if (strncmp(n, "LAYER_HEIGHT\0", 13) == 0) {
        current_layer->height = i;
    } else if (strncmp(n, "LAYER_CHANNELS\0", 15) == 0) {
        current_layer->channels = i;
    } else if (strncmp(n, "LAYER_PROBABILITY\0", 18) == 0) {
        current_layer->probability = f;
    } else if (strncmp(n, "LAYER_SIZE\0", 11) == 0) {
        current_layer->size = i;
    } else if (strncmp(n, "LAYER_STRIDE\0", 13) == 0) {
        current_layer->stride = i;
    } else if (strncmp(n, "LAYER_PAD\0", 10) == 0) {
        current_layer->pad = i;
    }
}

/**
 * @brief Sets classifier neural network condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 */
static void
config_cl_cond_neural(struct XCSF *xcsf, const char *n, const char *v)
{
    if (strncmp(n, "COND_LAYER_TYPE\0", 16) == 0) {
        if (xcsf->cond->largs == NULL) {
            xcsf->cond->largs = malloc(sizeof(struct ArgsLayer));
            current_layer = xcsf->cond->largs;
            layer_args_init(current_layer);
            current_layer->n_inputs = xcsf->x_dim;
        } else {
            struct ArgsLayer *tail = layer_args_tail(xcsf->cond->largs);
            tail->next = malloc(sizeof(struct ArgsLayer));
            current_layer = tail->next;
            layer_args_init(current_layer);
        }
        current_layer->n_init = 1; // single condition matching neuron
        if (xcsf->cond->type == RULE_TYPE_NEURAL) { // binary action outputs
            current_layer->n_init += (int) fmax(1, ceil(log2(xcsf->n_actions)));
        }
        current_layer->type = layer_type_as_int(v);
    }
}

/**
 * @brief Sets classifier DGP condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_cond_dgp(struct XCSF *xcsf, const char *n, const char *v, const int i,
                   const double f)
{
    (void) v;
    (void) f;
    if (strncmp(n, "COND_DGP_MAX_K\0", 15) == 0) {
        graph_param_set_max_k(xcsf->cond->dargs, i);
    } else if (strncmp(n, "COND_DGP_MAX_T\0", 15) == 0) {
        graph_param_set_max_t(xcsf->cond->dargs, i);
    } else if (strncmp(n, "COND_DGP_N\0", 11) == 0) {
        graph_param_set_n(xcsf->cond->dargs, i);
    } else if (strncmp(n, "COND_DGP_EVOLVE_CYCLES\0", 23) == 0) {
        graph_param_set_evolve_cycles(xcsf->cond->dargs, i);
    }
}

/**
 * @brief Sets classifier tree GP condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_cond_gp(struct XCSF *xcsf, const char *n, const char *v, const int i,
                  const double f)
{
    (void) v;
    if (strncmp(n, "COND_GP_NUM_CONS\0", 17) == 0) {
        tree_param_set_n_constants(xcsf->cond->targs, i);
    } else if (strncmp(n, "COND_GP_INIT_DEPTH\0", 19) == 0) {
        tree_param_set_init_depth(xcsf->cond->targs, i);
    } else if (strncmp(n, "COND_GP_MIN_CON\0", 16) == 0) {
        tree_param_set_min(xcsf->cond->targs, f);
    } else if (strncmp(n, "COND_GP_MAX_CON\0", 16) == 0) {
        tree_param_set_max(xcsf->cond->targs, f);
    } else if (strncmp(n, "COND_GP_MAX_LEN\0", 16) == 0) {
        tree_param_set_max_len(xcsf->cond->targs, i);
    }
}

/**
 * @brief Sets classifier center-spread-representation condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_cond_csr(struct XCSF *xcsf, const char *n, const char *v, const int i,
                   const double f)
{
    (void) i;
    (void) v;
    if (strncmp(n, "COND_MIN\0", 9) == 0) {
        cond_param_set_min(xcsf, f);
    } else if (strncmp(n, "COND_MAX\0", 9) == 0) {
        cond_param_set_max(xcsf, f);
    } else if (strncmp(n, "COND_SPREAD_MIN\0", 16) == 0) {
        cond_param_set_spread_min(xcsf, f);
    } else if (strncmp(n, "COND_ETA\0", 9) == 0) {
        cond_param_set_eta(xcsf, f);
    }
}

/**
 * @brief Sets classifier ternary condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_cond_ternary(struct XCSF *xcsf, const char *n, const char *v,
                       const int i, const double f)
{
    (void) v;
    if (strncmp(n, "COND_BITS\0", 10) == 0) {
        cond_param_set_bits(xcsf, i);
    } else if (strncmp(n, "COND_P_DONTCARE\0", 16) == 0) {
        cond_param_set_p_dontcare(xcsf, f);
    }
}

/**
 * @brief Sets classifier condition parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_cond(struct XCSF *xcsf, const char *n, const char *v, const int i,
               const double f)
{
    if (strncmp(n, "COND_TYPE\0", 10) == 0) {
        cond_param_set_type_string(xcsf, v);
    }
    config_cl_cond_ternary(xcsf, n, v, i, f);
    config_cl_cond_csr(xcsf, n, v, i, f);
    config_cl_cond_dgp(xcsf, n, v, i, f);
    config_cl_cond_gp(xcsf, n, v, i, f);
    config_cl_cond_neural(xcsf, n, v);
}

/**
 * @brief Sets classifier neural network prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 */
static void
config_cl_pred_neural(struct XCSF *xcsf, const char *n, const char *v)
{
    if (strncmp(n, "PRED_LAYER_TYPE\0", 16) == 0) {
        if (xcsf->pred->largs == NULL) {
            xcsf->pred->largs = malloc(sizeof(struct ArgsLayer));
            current_layer = xcsf->pred->largs;
            layer_args_init(current_layer);
            current_layer->n_inputs = xcsf->x_dim;
        } else {
            struct ArgsLayer *tail = layer_args_tail(xcsf->pred->largs);
            tail->next = malloc(sizeof(struct ArgsLayer));
            current_layer = tail->next;
            layer_args_init(current_layer);
            current_layer->n_inputs = tail->n_init;
        }
        current_layer->n_init = xcsf->y_dim;
        current_layer->type = layer_type_as_int(v);
    }
}

/**
 * @brief Sets classifier least squares prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_pred_ls(struct XCSF *xcsf, const char *n, const char *v, const int i,
                  const double f)
{
    (void) v;
    if (strncmp(n, "PRED_ETA\0", 9) == 0) {
        pred_param_set_eta(xcsf, f);
    } else if (strncmp(n, "PRED_ETA_MIN\0", 13) == 0) {
        pred_param_set_eta_min(xcsf, f);
    } else if (strncmp(n, "PRED_X0\0", 8) == 0) {
        pred_param_set_x0(xcsf, f);
    } else if (strncmp(n, "PRED_EVOLVE_ETA\0", 16) == 0) {
        pred_param_set_evolve_eta(xcsf, i);
    } else if (strncmp(n, "PRED_RLS_SCALE_FACTOR\0", 22) == 0) {
        pred_param_set_scale_factor(xcsf, f);
    } else if (strncmp(n, "PRED_RLS_LAMBDA\0", 16) == 0) {
        pred_param_set_lambda(xcsf, f);
    }
}

/**
 * @brief Sets classifier prediction parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_pred(struct XCSF *xcsf, const char *n, const char *v, const int i,
               const double f)
{
    if (strncmp(n, "PRED_TYPE\0", 10) == 0) {
        pred_param_set_type_string(xcsf, v);
    }
    config_cl_pred_ls(xcsf, n, v, i, f);
    config_cl_pred_neural(xcsf, n, v);
}

/**
 * @brief Sets classifier neural network action parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 */
static void
config_cl_act_neural(struct XCSF *xcsf, const char *n, const char *v)
{
    if (strncmp(n, "ACT_LAYER_TYPE\0", 15) == 0) {
        if (xcsf->act->largs == NULL) {
            xcsf->act->largs = malloc(sizeof(struct ArgsLayer));
            current_layer = xcsf->act->largs;
            layer_args_init(current_layer);
            current_layer->n_inputs = xcsf->x_dim;
        } else {
            struct ArgsLayer *tail = layer_args_tail(xcsf->act->largs);
            tail->next = malloc(sizeof(struct ArgsLayer));
            current_layer = tail->next;
            layer_args_init(current_layer);
            current_layer->n_inputs = tail->n_init;
        }
        current_layer->n_init = xcsf->n_actions;
        current_layer->type = layer_type_as_int(v);
    }
}

/**
 * @brief Sets classifier action parameters.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] n String representation of the parameter name.
 * @param [in] v String representation of the parameter value.
 * @param [in] i Integer representation of the parameter value.
 * @param [in] f Float representation of the parameter value.
 */
static void
config_cl_act(struct XCSF *xcsf, const char *n, const char *v, const int i,
              const double f)
{
    (void) i;
    (void) f;
    if (strncmp(n, "ACT_TYPE\0", 9) == 0) {
        action_param_set_type_string(xcsf, v);
    }
    config_cl_act_neural(xcsf, n, v);
}

/**
 * @brief Sets specified parameter.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] name Parameter name.
 * @param [in] value Parameter value.
 */
static void
config_add_param(struct XCSF *xcsf, const char *name, const char *value)
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
    const double f = strtod(value, &endptr);
    config_general(xcsf, name, value, i, f);
    config_multi(xcsf, name, value, i, f);
    config_subsump(xcsf, name, value, i, f);
    config_ea(xcsf, name, value, i, f);
    config_cl_gen(xcsf, name, value, i, f);
    config_cl_cond(xcsf, name, value, i, f);
    config_cl_pred(xcsf, name, value, i, f);
    config_cl_act(xcsf, name, value, i, f);
    if (current_layer != NULL) {
        config_layer(name, value, i, f);
    }
}

/**
 * @brief Removes tabs/spaces/lf/cr
 * @param [in] s The line to trim.
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
 * @param [in] xcsf The XCSF data structure.
 * @param [in] param The parameter to add.
 */
static void
config_newnvpair(struct XCSF *xcsf, const char *param)
{
    // get length of parameter name
    size_t namelen = 0;
    bool err = true;
    for (namelen = 0; namelen < strnlen(param, MAXLEN); ++namelen) {
        if (param[namelen] == '=') {
            err = false;
            break;
        }
    }
    if (err) {
        return; // no '=' found
    }
    // get parameter name
    char *name = malloc(namelen + 1);
    for (size_t i = 0; i < namelen; ++i) {
        name[i] = param[i];
    }
    name[namelen] = '\0';
    // get parameter value
    const size_t valuelen = strnlen(param, MAXLEN) - namelen;
    char *value = malloc(valuelen + 1);
    for (size_t i = 0; i < valuelen; ++i) {
        value[i] = param[namelen + 1 + i];
    }
    value[valuelen] = '\0';
    config_add_param(xcsf, name, value);
    free(name);
    free(value);
}

/**
 * @brief Parses a line of the config file and adds to the list.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] configline A single line of the configuration file.
 */
static void
config_process(struct XCSF *xcsf, const char *configline)
{
    if (strnlen(configline, MAXLEN) == 0) { // ignore empty lines
        return;
    }
    if (configline[0] == '#') { // lines starting with # are comments
        return;
    }
    char *ptr = strchr(configline, '#'); // remove anything after #
    if (ptr != NULL) {
        *ptr = '\0';
    }
    config_newnvpair(xcsf, configline);
}

/**
 * @brief Reads the specified configuration file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] filename The name of the configuration file.
 */
void
config_read(struct XCSF *xcsf, const char *filename)
{
    FILE *f = fopen(filename, "rt");
    if (f == NULL) {
        printf("Warning: could not open %s. %s.\n", filename, strerror(errno));
        return;
    }
    // clear existing layer parameters
    layer_args_free(&xcsf->act->largs);
    layer_args_free(&xcsf->cond->largs);
    layer_args_free(&xcsf->pred->largs);
    current_layer = NULL;
    // load new parameters from config
    char buff[MAXLEN];
    while (!feof(f)) {
        if (fgets(buff, MAXLEN - 2, f) == NULL) {
            break;
        }
        config_trim(buff);
        config_process(xcsf, buff);
    }
    fclose(f);
    layer_args_validate(xcsf->act->largs);
    layer_args_validate(xcsf->cond->largs);
    layer_args_validate(xcsf->pred->largs);
    tree_args_init_constants(xcsf->cond->targs);
}
