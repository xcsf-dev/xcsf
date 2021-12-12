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
 * @file pybind_wrapper.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @author David Pätzel
 * @copyright The Authors.
 * @date 2020--2021.
 * @brief Python library wrapper functions.
 */

#ifdef _WIN32 // Try to work around https://bugs.python.org/issue11566
    #define _hypot hypot
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace py = pybind11;

extern "C" {
#include "action.h"
#include "clset.h"
#include "clset_neural.h"
#include "condition.h"
#include "dgp.h"
#include "ea.h"
#include "gp.h"
#include "neural_activations.h"
#include "neural_layer.h"
#include "pa.h"
#include "param.h"
#include "prediction.h"
#include "utils.h"
#include "xcs_rl.h"
#include "xcs_supervised.h"
}

/**
 * @brief Python XCSF class data structure.
 */
class XCS
{
  private:
    struct XCSF xcs; //!< XCSF data structure
    double *state; //!< Current input state for RL
    int action; //!< Current action for RL
    double payoff; //!< Current reward for RL
    struct Input *train_data; //!< Training data for supervised learning
    struct Input *test_data; //!< Test data for supervised learning

  public:
    /**
     * @brief Constructor.
     * @param [in] x_dim The dimensionality of the input variables.
     * @param [in] y_dim The dimensionality of the prediction variables.
     * @param [in] n_actions The total number of possible actions.
     */
    XCS(const int x_dim, const int y_dim, const int n_actions)
    {
        param_init(&xcs, x_dim, y_dim, n_actions);
        xcsf_init(&xcs);
        pa_init(&xcs);
        state = NULL;
        action = 0;
        payoff = 0;
        train_data = (struct Input *) malloc(sizeof(struct Input));
        train_data->n_samples = 0;
        train_data->x_dim = 0;
        train_data->y_dim = 0;
        train_data->x = NULL;
        train_data->y = NULL;
        test_data = (struct Input *) malloc(sizeof(struct Input));
        test_data->n_samples = 0;
        test_data->x_dim = 0;
        test_data->y_dim = 0;
        test_data->x = NULL;
        test_data->y = NULL;
    }

    /**
     * @brief Returns the XCSF major version number.
     * @return Major version number.
     */
    int
    version_major(void)
    {
        return VERSION_MAJOR;
    }

    /**
     * @brief Returns the XCSF minor version number.
     * @return Minor version number.
     */
    int
    version_minor(void)
    {
        return VERSION_MINOR;
    }

    /**
     * @brief Returns the XCSF build version number.
     * @return Build version number.
     */
    int
    version_build(void)
    {
        return VERSION_BUILD;
    }

    /**
     * @brief Writes the entire current state of XCSF to a file.
     * @param [in] filename String containing the name of the output file.
     * @return The total number of elements written.
     */
    size_t
    save(const char *filename)
    {
        return xcsf_save(&xcs, filename);
    }

    /**
     * @brief Reads the entire current state of XCSF from a file.
     * @param [in] filename String containing the name of the input file.
     * @return The total number of elements read.
     */
    size_t
    load(const char *filename)
    {
        return xcsf_load(&xcs, filename);
    }

    /**
     * @brief Stores the current population in memory for later retrieval.
     */
    void
    store(void)
    {
        xcsf_store_pset(&xcs);
    }

    /**
     * @brief Retrieves the stored population, setting it as current.
     */
    void
    retrieve(void)
    {
        xcsf_retrieve_pset(&xcs);
    }

    /**
     * @brief Prints the XCSF parameters and their current values.
     */
    void
    print_params(void)
    {
        param_print(&xcs);
    }

    /**
     * @brief Inserts a new hidden layer before the output layer within all
     * prediction neural networks in the population.
     */
    void
    pred_expand(void)
    {
        xcsf_pred_expand(&xcs);
    }

    /**
     * @brief Switches from autoencoding to classification.
     * @param [in] y_dim The output dimension (i.e., the number of classes).
     * @param [in] n_del The number of hidden layers to remove.
     */
    void
    ae_to_classifier(const int y_dim, const int n_del)
    {
        xcsf_ae_to_classifier(&xcs, y_dim, n_del);
    }

    /**
     * @brief Prints the current population.
     * @param [in] print_cond Whether to print the condition.
     * @param [in] print_act Whether to print the action.
     * @param [in] print_pred Whether to print the prediction.
     */
    void
    print_pset(const bool print_cond, const bool print_act,
               const bool print_pred)
    {
        xcsf_print_pset(&xcs, print_cond, print_act, print_pred);
    }

    /**
     * @brief Returns a JSON formatted string representing the population set.
     * @param [in] return_cond Whether to return the condition.
     * @param [in] return_act Whether to return the action.
     * @param [in] return_pred Whether to return the prediction.
     * @return String encoded in json format.
     */
    const char *
    json_export(const bool return_cond, const bool return_act,
                const bool return_pred)
    {
        if (xcs.pset.list != NULL) {
            return clset_json_export(&xcs, &xcs.pset, return_cond, return_act,
                                     return_pred);
        }
        return "null";
    }

    /**
     * @brief Returns a JSON formatted string representing the parameters.
     * @return String encoded in json format.
     */
    const char *
    json_parameters()
    {
        return param_json_export(&xcs);
    }

    /* Reinforcement learning */

    /**
     * @brief Creates/updates an action set for a given (state, action, reward).
     * @param [in] input The input state to match.
     * @param [in] action The selected action.
     * @param [in] reward The reward for having performed the action.
     * @return The prediction error.
     */
    double
    fit(const py::array_t<double> input, const int action, const double reward)
    {
        py::buffer_info buf = input.request();
        if (buf.shape[0] != xcs.x_dim) {
            printf("fit() error: x_dim is not equal to: %d.\n", xcs.x_dim);
            exit(EXIT_FAILURE);
        }
        if (action < 0 || action >= xcs.n_actions) {
            printf("fit() error: action outside: [0,%d).\n", xcs.n_actions);
            exit(EXIT_FAILURE);
        }
        state = (double *) buf.ptr;
        return xcs_rl_fit(&xcs, state, action, reward);
    }

    /**
     * @brief Initialises a reinforcement learning trial.
     */
    void
    init_trial(void)
    {
        if (xcs.time == 0) {
            clset_pset_init(&xcs);
        }
        xcs_rl_init_trial(&xcs);
    }

    /**
     * @brief Frees memory used by a reinforcement learning trial.
     */
    void
    end_trial(void)
    {
        xcs_rl_end_trial(&xcs);
    }

    /**
     * @brief Initialises a step in a reinforcement learning trial.
     */
    void
    init_step(void)
    {
        xcs_rl_init_step(&xcs);
    }

    /**
     * @brief Ends a step in a reinforcement learning trial.
     */
    void
    end_step(void)
    {
        xcs_rl_end_step(&xcs, state, action, payoff);
    }

    /**
     * @brief Selects an action to perform in a reinforcement learning problem.
     * @details Constructs the match set and selects an action to perform.
     * @param [in] input The input state.
     * @param [in] explore Whether this is an exploration step.
     * @return The selected action.
     */
    int
    decision(const py::array_t<double> input, const bool explore)
    {
        py::buffer_info buf = input.request();
        if (buf.shape[0] != xcs.x_dim) {
            printf("decision() error: x_dim is not equal to: %d.\n", xcs.x_dim);
            exit(EXIT_FAILURE);
        }
        state = (double *) buf.ptr;
        param_set_explore(&xcs, explore);
        action = xcs_rl_decision(&xcs, state);
        return action;
    }

    /**
     * @brief Creates the action set using the previously selected action,
     * updates the classifiers, and runs the EA on explore steps.
     * @param [in] reward The reward from performing the action.
     * @param [in] done Whether the environment is in a terminal state.
     */
    void
    update(const double reward, const bool done)
    {
        payoff = reward;
        xcs_rl_update(&xcs, state, action, payoff, done);
    }

    /**
     * @brief Returns the reinforcement learning system prediction error.
     * @param [in] reward The current reward.
     * @param [in] done Whether the environment is in a terminal state.
     * @param [in] max_p The maximum payoff in the environment.
     * @return The prediction error.
     */
    double
    error(const double reward, const bool done, const double max_p)
    {
        payoff = reward;
        return xcs_rl_error(&xcs, action, payoff, done, max_p);
    }

    /* Supervised learning */

    /**
     * @brief Loads an input data structure for fitting.
     * @param [in,out] data Input data structure used to point to the data.
     * @param [in] X Vector of features with shape (n_samples, x_dim).
     * @param [in] Y Vector of truth values with shape (n_samples, y_dim).
     */
    void
    load_input(struct Input *data, const py::array_t<double> X,
               const py::array_t<double> Y)
    {
        const py::buffer_info buf_x = X.request();
        const py::buffer_info buf_y = Y.request();
        if (buf_x.shape[0] != buf_y.shape[0]) {
            printf("load_input() error: X and Y n_samples are not equal.\n");
            exit(EXIT_FAILURE);
        }
        if (buf_x.shape[1] != xcs.x_dim) {
            printf("load_input() error: x_dim != %d.\n", xcs.x_dim);
            printf("2-D arrays are required. Perhaps reshape your data.\n");
            exit(EXIT_FAILURE);
        }
        if (buf_y.shape[1] != xcs.y_dim) {
            printf("load_input() error: y_dim != %d.\n", xcs.y_dim);
            printf("2-D arrays are required. Perhaps reshape your data.\n");
            exit(EXIT_FAILURE);
        }
        data->n_samples = buf_x.shape[0];
        data->x_dim = buf_x.shape[1];
        data->y_dim = buf_y.shape[1];
        data->x = (double *) buf_x.ptr;
        data->y = (double *) buf_y.ptr;
    }

    /**
     * @brief Executes MAX_TRIALS number of XCSF learning iterations using the
     * provided training data.
     * @param [in] train_X The input values to use for training.
     * @param [in] train_Y The true output values to use for training.
     * @param [in] shuffle Whether to randomise the instances during training.
     * @return The average XCSF training error using the loss function.
     */
    double
    fit(const py::array_t<double> train_X, const py::array_t<double> train_Y,
        const bool shuffle)
    {
        load_input(train_data, train_X, train_Y);
        if (xcs.time == 0) { // first execution
            clset_pset_init(&xcs);
        }
        return xcs_supervised_fit(&xcs, train_data, NULL, shuffle);
    }

    /**
     * @brief Executes MAX_TRIALS number of XCSF learning iterations using the
     * provided training data and test iterations using the test data.
     * @param [in] train_X The input values to use for training.
     * @param [in] train_Y The true output values to use for training.
     * @param [in] test_X The input values to use for testing.
     * @param [in] test_Y The true output values to use for testing.
     * @param [in] shuffle Whether to randomise the instances during training.
     * @return The average XCSF training error using the loss function.
     */
    double
    fit(const py::array_t<double> train_X, const py::array_t<double> train_Y,
        const py::array_t<double> test_X, const py::array_t<double> test_Y,
        const bool shuffle)
    {
        load_input(train_data, train_X, train_Y);
        load_input(test_data, test_X, test_Y);
        if (xcs.time == 0) { // first execution
            clset_pset_init(&xcs);
        }
        return xcs_supervised_fit(&xcs, train_data, test_data, shuffle);
    }

    /**
     * @brief Returns the XCSF prediction array for the provided input.
     * @param [in] X The input variables.
     * @return The prediction array values.
     */
    py::array_t<double>
    predict(const py::array_t<double> X)
    {
        const py::buffer_info buf_x = X.request();
        const int n_samples = buf_x.shape[0];
        if (buf_x.shape[1] != xcs.x_dim) {
            printf("predict() error: x_dim is not equal to: %d.\n", xcs.x_dim);
            printf("2-D arrays are required. Perhaps reshape your data.\n");
            exit(EXIT_FAILURE);
        }
        const double *input = (double *) buf_x.ptr;
        double *output =
            (double *) malloc(sizeof(double) * n_samples * xcs.pa_size);
        xcs_supervised_predict(&xcs, input, output, n_samples);
        return py::array_t<double>(
            std::vector<ptrdiff_t>{ n_samples, xcs.pa_size }, output);
    }

    /**
     * @brief Returns the error over one sequential pass of the provided data.
     * @param [in] X The input values to use for scoring.
     * @param [in] Y The true output values to use for scoring.
     * @return The average XCSF error using the loss function.
     */
    double
    score(const py::array_t<double> X, const py::array_t<double> Y)
    {
        return score(X, Y, 0);
    }

    /**
     * @brief Returns the error using N random samples from the provided data.
     * @param [in] X The input values to use for scoring.
     * @param [in] Y The true output values to use for scoring.
     * @param [in] N The maximum number of samples to draw randomly for scoring.
     * @return The average XCSF error using the loss function.
     */
    double
    score(const py::array_t<double> X, const py::array_t<double> Y, const int N)
    {
        load_input(test_data, X, Y);
        if (N > 1) {
            return xcs_supervised_score_n(&xcs, test_data, N);
        }
        return xcs_supervised_score(&xcs, test_data);
    }

    /* GETTERS */

    /**
     * @brief Returns the current system error.
     * @return Moving average of the system error, updated with step size BETA.
     */
    double
    error(void)
    {
        return xcs.error;
    }

    int
    get_omp_num_threads(void)
    {
        return xcs.OMP_NUM_THREADS;
    }

    bool
    get_pop_init(void)
    {
        return xcs.POP_INIT;
    }

    int
    get_max_trials(void)
    {
        return xcs.MAX_TRIALS;
    }

    int
    get_perf_trials(void)
    {
        return xcs.PERF_TRIALS;
    }

    int
    get_pop_max_size(void)
    {
        return xcs.POP_SIZE;
    }

    const char *
    get_loss_func(void)
    {
        return loss_type_as_string(xcs.LOSS_FUNC);
    }

    double
    get_huber_delta(void)
    {
        return xcs.HUBER_DELTA;
    }

    double
    get_alpha(void)
    {
        return xcs.ALPHA;
    }

    double
    get_beta(void)
    {
        return xcs.BETA;
    }

    double
    get_delta(void)
    {
        return xcs.DELTA;
    }

    double
    get_e0(void)
    {
        return xcs.E0;
    }

    double
    get_init_error(void)
    {
        return xcs.INIT_ERROR;
    }

    double
    get_init_fitness(void)
    {
        return xcs.INIT_FITNESS;
    }

    double
    get_nu(void)
    {
        return xcs.NU;
    }

    int
    get_m_probation(void)
    {
        return xcs.M_PROBATION;
    }

    bool
    get_stateful(void)
    {
        return xcs.STATEFUL;
    }

    bool
    get_compaction(void)
    {
        return xcs.COMPACTION;
    }

    int
    get_theta_del(void)
    {
        return xcs.THETA_DEL;
    }

    int
    get_theta_sub(void)
    {
        return xcs.THETA_SUB;
    }

    bool
    get_set_subsumption(void)
    {
        return xcs.SET_SUBSUMPTION;
    }

    int
    get_pset_size(void)
    {
        return xcs.pset.size;
    }

    int
    get_pset_num(void)
    {
        return xcs.pset.num;
    }

    int
    get_time(void)
    {
        return xcs.time;
    }

    int
    get_x_dim(void)
    {
        return xcs.x_dim;
    }

    int
    get_y_dim(void)
    {
        return xcs.y_dim;
    }

    int
    get_n_actions(void)
    {
        return xcs.n_actions;
    }

    double
    get_pset_mean_cond_size(void)
    {
        return clset_mean_cond_size(&xcs, &xcs.pset);
    }

    double
    get_pset_mean_pred_size(void)
    {
        return clset_mean_pred_size(&xcs, &xcs.pset);
    }

    double
    get_pset_mean_pred_eta(const int layer)
    {
        return clset_mean_pred_eta(&xcs, &xcs.pset, layer);
    }

    double
    get_pset_mean_pred_neurons(const int layer)
    {
        return clset_mean_pred_neurons(&xcs, &xcs.pset, layer);
    }

    double
    get_pset_mean_pred_connections(const int layer)
    {
        return clset_mean_pred_connections(&xcs, &xcs.pset, layer);
    }

    double
    get_pset_mean_pred_layers(void)
    {
        return clset_mean_pred_layers(&xcs, &xcs.pset);
    }

    double
    get_pset_mean_cond_connections(const int layer)
    {
        return clset_mean_cond_connections(&xcs, &xcs.pset, layer);
    }

    double
    get_pset_mean_cond_neurons(const int layer)
    {
        return clset_mean_cond_neurons(&xcs, &xcs.pset, layer);
    }

    double
    get_pset_mean_cond_layers(void)
    {
        return clset_mean_cond_layers(&xcs, &xcs.pset);
    }

    double
    get_mset_size(void)
    {
        return xcs.mset_size;
    }

    double
    get_aset_size(void)
    {
        return xcs.aset_size;
    }

    double
    get_mfrac(void)
    {
        return xcs.mfrac;
    }

    int
    get_teletransportation(void)
    {
        return xcs.TELETRANSPORTATION;
    }

    double
    get_gamma(void)
    {
        return xcs.GAMMA;
    }

    double
    get_p_explore(void)
    {
        return xcs.P_EXPLORE;
    }

    const char *
    get_ea_select_type(void)
    {
        return ea_type_as_string(xcs.ea->select_type);
    }

    double
    get_ea_select_size(void)
    {
        return xcs.ea->select_size;
    }

    double
    get_theta_ea(void)
    {
        return xcs.ea->theta;
    }

    int
    get_lambda(void)
    {
        return xcs.ea->lambda;
    }

    double
    get_p_crossover(void)
    {
        return xcs.ea->p_crossover;
    }

    double
    get_err_reduc(void)
    {
        return xcs.ea->err_reduc;
    }

    double
    get_fit_reduc(void)
    {
        return xcs.ea->fit_reduc;
    }

    bool
    get_ea_subsumption(void)
    {
        return xcs.ea->subsumption;
    }

    bool
    get_ea_pred_reset(void)
    {
        return xcs.ea->pred_reset;
    }

    /* SETTERS */

    /**
     * @brief Sets the condition type.
     * @param [in] type String representing a name of a condition type.
     */
    void
    set_condition(const std::string &type)
    {
        cond_param_set_type_string(&xcs, type.c_str());
    }

    /**
     * @brief Sets the action type.
     * @param [in] type String representing a name of an action type.
     */
    void
    set_action(const std::string &type)
    {
        action_param_set_type_string(&xcs, type.c_str());
    }

    /**
     * @brief Sets the prediction type.
     * @param [in] type String representing a name of a prediction type.
     */
    void
    set_prediction(const std::string &type)
    {
        pred_param_set_type_string(&xcs, type.c_str());
    }

    /**
     * @brief Sets the condition type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    set_condition(const std::string &type, const py::dict &args)
    {
        cond_param_set_type_string(&xcs, type.c_str());
        switch (xcs.cond->type) {
            case COND_TYPE_HYPERRECTANGLE:
            case COND_TYPE_HYPERELLIPSOID:
                unpack_cond_csr(args);
                break;
            case COND_TYPE_NEURAL:
            case RULE_TYPE_NEURAL:
            case RULE_TYPE_NETWORK:
                unpack_cond_neural(args);
                break;
            case COND_TYPE_GP:
                unpack_cond_gp(args);
                break;
            case COND_TYPE_DGP:
            case RULE_TYPE_DGP:
                unpack_cond_dgp(args);
                break;
            case COND_TYPE_TERNARY:
                unpack_cond_ternary(args);
                break;
            default:
                break;
        }
    }

    /**
     * @brief Sets parameters used by center-spread conditions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_cond_csr(const py::dict &args)
    {
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "min") {
                cond_param_set_min(&xcs, item.second.cast<double>());
            } else if (name == "max") {
                cond_param_set_max(&xcs, item.second.cast<double>());
            } else if (name == "spread_min") {
                cond_param_set_spread_min(&xcs, item.second.cast<double>());
            } else if (name == "eta") {
                cond_param_set_eta(&xcs, item.second.cast<double>());
            } else {
                printf("Unknown center-spread parameter: %s\n", name.c_str());
                exit(EXIT_FAILURE);
            }
        }
    }

    /**
     * @brief Sets parameters used by tree-GP conditions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_cond_gp(const py::dict &args)
    {
        struct ArgsGPTree *targs = xcs.cond->targs;
        tree_param_set_n_inputs(targs, xcs.x_dim);
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "n_constants") {
                tree_param_set_n_constants(targs, item.second.cast<int>());
            } else if (name == "init_depth") {
                tree_param_set_init_depth(targs, item.second.cast<int>());
            } else if (name == "max_len") {
                tree_param_set_max_len(targs, item.second.cast<int>());
            } else if (name == "min") {
                tree_param_set_min(targs, item.second.cast<double>());
            } else if (name == "max") {
                tree_param_set_max(targs, item.second.cast<double>());
            } else {
                printf("Unknown tree-GP parameter: %s\n", name.c_str());
                exit(EXIT_FAILURE);
            }
        }
        tree_args_init_constants(targs);
    }

    /**
     * @brief Sets parameters used by dynamical GP graph conditions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_cond_dgp(const py::dict &args)
    {
        struct ArgsDGP *dargs = xcs.cond->dargs;
        graph_param_set_n_inputs(dargs, xcs.x_dim);
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "max_k") {
                graph_param_set_max_k(dargs, item.second.cast<int>());
            } else if (name == "max_t") {
                graph_param_set_max_t(dargs, item.second.cast<int>());
            } else if (name == "n") {
                graph_param_set_n(dargs, item.second.cast<int>());
            } else if (name == "evolve_cycles") {
                graph_param_set_evolve_cycles(dargs, item.second.cast<bool>());
            } else {
                printf("Unknown DGP parameter: %s\n", name.c_str());
                exit(EXIT_FAILURE);
            }
        }
    }

    /**
     * @brief Sets parameters used by ternary conditions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_cond_ternary(const py::dict &args)
    {
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "bits") {
                cond_param_set_bits(&xcs, item.second.cast<int>());
            } else if (name == "p_dontcare") {
                cond_param_set_p_dontcare(&xcs, item.second.cast<double>());
            } else {
                printf("Unknown ternary parameter: %s\n", name.c_str());
                exit(EXIT_FAILURE);
            }
        }
    }

    /**
     * @brief Sets parameters used by neural network conditions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_cond_neural(const py::dict &args)
    {
        layer_args_free(&xcs.cond->largs);
        for (auto item : args) {
            struct ArgsLayer *larg =
                (struct ArgsLayer *) malloc(sizeof(struct ArgsLayer));
            layer_args_init(larg);
            unpack_layer_params(larg, item.second.cast<py::dict>());
            if (xcs.cond->largs == NULL) {
                xcs.cond->largs = larg;
            } else {
                struct ArgsLayer *iter = xcs.cond->largs;
                while (iter->next != NULL) {
                    iter = iter->next;
                }
                iter->next = larg;
            }
        }
        layer_args_validate(xcs.cond->largs);
    }

    /**
     * @brief Sets parameters used by a neural network layer.
     * @param [in] larg Layer parameter structure to set.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_layer_params(struct ArgsLayer *larg, const py::dict &args)
    {
        larg->n_inputs = xcs.x_dim;
        for (std::pair<py::handle, py::handle> item : args) {
            const auto name = item.first.cast<std::string>();
            if (name == "type") {
                const auto value = item.second.cast<std::string>();
                larg->type = layer_type_as_int(value.c_str());
            } else if (name == "max_neuron_grow") {
                larg->max_neuron_grow = item.second.cast<int>();
            } else if (name == "evolve_weights") {
                larg->evolve_weights = item.second.cast<bool>();
            } else if (name == "evolve_neurons") {
                larg->evolve_neurons = item.second.cast<bool>();
            } else if (name == "evolve_functions") {
                larg->evolve_functions = item.second.cast<bool>();
            } else if (name == "evolve_connect") {
                larg->evolve_connect = item.second.cast<bool>();
            } else if (name == "evolve_eta") {
                larg->evolve_eta = item.second.cast<bool>();
            } else if (name == "sgd_weights") {
                larg->sgd_weights = item.second.cast<bool>();
            } else if (name == "activation") {
                const auto value = item.second.cast<std::string>();
                larg->function = neural_activation_as_int(value.c_str());
            } else if (name == "recurrent_activation") {
                const auto value = item.second.cast<std::string>();
                larg->recurrent_function =
                    neural_activation_as_int(value.c_str());
            } else if (name == "n_init") {
                larg->n_init = item.second.cast<int>();
            } else if (name == "n_max") {
                larg->n_max = item.second.cast<int>();
            } else if (name == "eta") {
                larg->eta = item.second.cast<double>();
            } else if (name == "eta_min") {
                larg->eta_min = item.second.cast<double>();
            } else if (name == "momentum") {
                larg->momentum = item.second.cast<double>();
            } else if (name == "decay") {
                larg->decay = item.second.cast<double>();
            } else if (name == "scale") {
                larg->scale = item.second.cast<double>();
            } else if (name == "probability") {
                larg->probability = item.second.cast<double>();
            } else if (name == "height") {
                larg->height = item.second.cast<int>();
            } else if (name == "width") {
                larg->width = item.second.cast<int>();
            } else if (name == "channels") {
                larg->channels = item.second.cast<int>();
            } else if (name == "size") {
                larg->size = item.second.cast<int>();
            } else if (name == "stride") {
                larg->stride = item.second.cast<int>();
            } else if (name == "pad") {
                larg->pad = item.second.cast<int>();
            } else {
                printf("Unknown neural layer parameter: %s\n", name.c_str());
                exit(EXIT_FAILURE);
            }
        }
    }

    /**
     * @brief Sets the action type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    set_action(const std::string &type, const py::dict &args)
    {
        action_param_set_type_string(&xcs, type.c_str());
        if (xcs.act->type == ACT_TYPE_NEURAL) {
            unpack_act_neural(args);
        }
    }

    /**
     * @brief Sets parameters used by neural network actions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_act_neural(const py::dict &args)
    {
        layer_args_free(&xcs.act->largs);
        for (auto item : args) {
            struct ArgsLayer *larg =
                (struct ArgsLayer *) malloc(sizeof(struct ArgsLayer));
            layer_args_init(larg);
            unpack_layer_params(larg, item.second.cast<py::dict>());
            if (xcs.act->largs == NULL) {
                xcs.act->largs = larg;
            } else {
                struct ArgsLayer *iter = xcs.act->largs;
                while (iter->next != NULL) {
                    iter = iter->next;
                }
                iter->next = larg;
            }
        }
        layer_args_validate(xcs.act->largs);
    }

    /**
     * @brief Sets the prediction type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    set_prediction(const std::string &type, const py::dict &args)
    {
        pred_param_set_type_string(&xcs, type.c_str());
        switch (xcs.pred->type) {
            case PRED_TYPE_NLMS_LINEAR:
            case PRED_TYPE_NLMS_QUADRATIC:
                unpack_pred_nlms(args);
                break;
            case PRED_TYPE_RLS_LINEAR:
            case PRED_TYPE_RLS_QUADRATIC:
                unpack_pred_rls(args);
                break;
            case PRED_TYPE_NEURAL:
                unpack_pred_neural(args);
                break;
            default:
                break;
        }
    }

    /**
     * @brief Sets parameters used by neural network predictions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_pred_neural(const py::dict &args)
    {
        layer_args_free(&xcs.pred->largs);
        for (auto item : args) {
            struct ArgsLayer *larg =
                (struct ArgsLayer *) malloc(sizeof(struct ArgsLayer));
            layer_args_init(larg);
            unpack_layer_params(larg, item.second.cast<py::dict>());
            if (xcs.pred->largs == NULL) {
                xcs.pred->largs = larg;
            } else {
                struct ArgsLayer *iter = xcs.pred->largs;
                while (iter->next != NULL) {
                    iter = iter->next;
                }
                iter->next = larg;
            }
        }
        layer_args_validate(xcs.pred->largs);
    }

    /**
     * @brief Sets parameters used by least mean squares predictions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_pred_nlms(const py::dict &args)
    {
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "x0") {
                pred_param_set_x0(&xcs, item.second.cast<double>());
            } else if (name == "eta") {
                pred_param_set_eta(&xcs, item.second.cast<double>());
            } else if (name == "eta_min") {
                pred_param_set_eta_min(&xcs, item.second.cast<double>());
            } else if (name == "evolve_eta") {
                pred_param_set_evolve_eta(&xcs, item.second.cast<bool>());
            } else {
                printf("Unknown NLMS parameter: %s\n", name.c_str());
                exit(EXIT_FAILURE);
            }
        }
    }

    /**
     * @brief Sets parameters used by recursive least mean squares predictions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_pred_rls(const py::dict &args)
    {
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "x0") {
                pred_param_set_x0(&xcs, item.second.cast<double>());
            } else if (name == "rls_scale_factor") {
                pred_param_set_scale_factor(&xcs, item.second.cast<double>());
            } else if (name == "rls_lambda") {
                pred_param_set_lambda(&xcs, item.second.cast<double>());
            } else {
                printf("Unknown RLS parameter: %s\n", name.c_str());
                exit(EXIT_FAILURE);
            }
        }
    }

    void
    set_omp_num_threads(const int a)
    {
        param_set_omp_num_threads(&xcs, a);
    }

    void
    set_pop_init(const bool a)
    {
        param_set_pop_init(&xcs, a);
    }

    void
    set_max_trials(const int a)
    {
        param_set_max_trials(&xcs, a);
    }

    void
    set_perf_trials(const int a)
    {
        param_set_perf_trials(&xcs, a);
    }

    void
    set_pop_max_size(const int a)
    {
        param_set_pop_size(&xcs, a);
    }

    void
    set_loss_func(const char *a)
    {
        param_set_loss_func_string(&xcs, a);
    }

    void
    set_huber_delta(const double a)
    {
        param_set_huber_delta(&xcs, a);
    }

    void
    set_alpha(const double a)
    {
        param_set_alpha(&xcs, a);
    }

    void
    set_beta(const double a)
    {
        param_set_beta(&xcs, a);
    }

    void
    set_delta(const double a)
    {
        param_set_delta(&xcs, a);
    }

    void
    set_e0(const double a)
    {
        param_set_e0(&xcs, a);
    }

    void
    set_init_error(const double a)
    {
        param_set_init_error(&xcs, a);
    }

    void
    set_init_fitness(const double a)
    {
        param_set_init_fitness(&xcs, a);
    }

    void
    set_nu(const double a)
    {
        param_set_nu(&xcs, a);
    }

    void
    set_m_probation(const int a)
    {
        param_set_m_probation(&xcs, a);
    }

    void
    set_theta_del(const int a)
    {
        param_set_theta_del(&xcs, a);
    }

    void
    set_theta_sub(const int a)
    {
        param_set_theta_sub(&xcs, a);
    }

    void
    set_set_subsumption(const bool a)
    {
        param_set_set_subsumption(&xcs, a);
    }

    void
    set_teletransportation(const int a)
    {
        param_set_teletransportation(&xcs, a);
    }

    void
    set_stateful(const bool a)
    {
        param_set_stateful(&xcs, a);
    }

    void
    set_compaction(const bool a)
    {
        param_set_compaction(&xcs, a);
    }

    void
    set_gamma(const double a)
    {
        param_set_gamma(&xcs, a);
    }

    void
    set_p_explore(const double a)
    {
        param_set_p_explore(&xcs, a);
    }

    void
    set_ea_select_type(const char *a)
    {
        ea_param_set_type_string(&xcs, a);
    }

    void
    set_ea_select_size(const double a)
    {
        ea_param_set_select_size(&xcs, a);
    }

    void
    set_theta_ea(const double a)
    {
        ea_param_set_theta(&xcs, a);
    }

    void
    set_lambda(const int a)
    {
        ea_param_set_lambda(&xcs, a);
    }

    void
    set_p_crossover(const double a)
    {
        ea_param_set_p_crossover(&xcs, a);
    }

    void
    set_err_reduc(const double a)
    {
        ea_param_set_err_reduc(&xcs, a);
    }

    void
    set_fit_reduc(const double a)
    {
        ea_param_set_fit_reduc(&xcs, a);
    }

    void
    set_ea_subsumption(const bool a)
    {
        ea_param_set_subsumption(&xcs, a);
    }

    void
    set_ea_pred_reset(const bool a)
    {
        ea_param_set_pred_reset(&xcs, a);
    }

    void
    seed(const uint32_t seed)
    {
        rand_init_seed(seed);
    }
};

PYBIND11_MODULE(xcsf, m)
{
    rand_init();

    double (XCS::*fit1)(const py::array_t<double>, const int, const double) =
        &XCS::fit;
    double (XCS::*fit2)(const py::array_t<double>, const py::array_t<double>,
                        const bool) = &XCS::fit;
    double (XCS::*fit3)(const py::array_t<double>, const py::array_t<double>,
                        const py::array_t<double>, const py::array_t<double>,
                        const bool) = &XCS::fit;

    double (XCS::*score1)(const py::array_t<double> test_X,
                          const py::array_t<double> test_Y) = &XCS::score;
    double (XCS::*score2)(const py::array_t<double> test_X,
                          const py::array_t<double> test_Y, const int N) =
        &XCS::score;

    double (XCS::*error1)(void) = &XCS::error;
    double (XCS::*error2)(const double, const bool, const double) = &XCS::error;

    void (XCS::*condition1)(const std::string &) = &XCS::set_condition;
    void (XCS::*condition2)(const std::string &, const py::dict &) =
        &XCS::set_condition;

    void (XCS::*action1)(const std::string &) = &XCS::set_action;
    void (XCS::*action2)(const std::string &, const py::dict &) =
        &XCS::set_action;

    void (XCS::*prediction1)(const std::string &) = &XCS::set_prediction;
    void (XCS::*prediction2)(const std::string &, const py::dict &) =
        &XCS::set_prediction;

    py::class_<XCS>(m, "XCS")
        .def(py::init<const int, const int, const int>())
        .def("condition", condition1)
        .def("condition", condition2)
        .def("action", action1)
        .def("action", action2)
        .def("prediction", prediction1)
        .def("prediction", prediction2)
        .def("fit", fit1)
        .def("fit", fit2)
        .def("fit", fit3)
        .def("score", score1)
        .def("score", score2)
        .def("error", error1)
        .def("error", error2)
        .def("predict", &XCS::predict)
        .def("save", &XCS::save)
        .def("load", &XCS::load)
        .def("store", &XCS::store)
        .def("retrieve", &XCS::retrieve)
        .def("version_major", &XCS::version_major)
        .def("version_minor", &XCS::version_minor)
        .def("version_build", &XCS::version_build)
        .def("init_trial", &XCS::init_trial)
        .def("end_trial", &XCS::end_trial)
        .def("init_step", &XCS::init_step)
        .def("end_step", &XCS::end_step)
        .def("decision", &XCS::decision)
        .def("update", &XCS::update)
        .def("seed", &XCS::seed)
        .def_property("OMP_NUM_THREADS", &XCS::get_omp_num_threads,
                      &XCS::set_omp_num_threads)
        .def_property("POP_INIT", &XCS::get_pop_init, &XCS::set_pop_init)
        .def_property("MAX_TRIALS", &XCS::get_max_trials, &XCS::set_max_trials)
        .def_property("PERF_TRIALS", &XCS::get_perf_trials,
                      &XCS::set_perf_trials)
        .def_property("POP_SIZE", &XCS::get_pop_max_size,
                      &XCS::set_pop_max_size)
        .def_property("LOSS_FUNC", &XCS::get_loss_func, &XCS::set_loss_func)
        .def_property("HUBER_DELTA", &XCS::get_huber_delta,
                      &XCS::set_huber_delta)
        .def_property("ALPHA", &XCS::get_alpha, &XCS::set_alpha)
        .def_property("BETA", &XCS::get_beta, &XCS::set_beta)
        .def_property("DELTA", &XCS::get_delta, &XCS::set_delta)
        .def_property("E0", &XCS::get_e0, &XCS::set_e0)
        .def_property("STATEFUL", &XCS::get_stateful, &XCS::set_stateful)
        .def_property("COMPACTION", &XCS::get_compaction, &XCS::set_compaction)
        .def_property("INIT_ERROR", &XCS::get_init_error, &XCS::set_init_error)
        .def_property("INIT_FITNESS", &XCS::get_init_fitness,
                      &XCS::set_init_fitness)
        .def_property("NU", &XCS::get_nu, &XCS::set_nu)
        .def_property("M_PROBATION", &XCS::get_m_probation,
                      &XCS::set_m_probation)
        .def_property("THETA_DEL", &XCS::get_theta_del, &XCS::set_theta_del)
        .def_property("THETA_SUB", &XCS::get_theta_sub, &XCS::set_theta_sub)
        .def_property("SET_SUBSUMPTION", &XCS::get_set_subsumption,
                      &XCS::set_set_subsumption)
        .def_property("TELETRANSPORTATION", &XCS::get_teletransportation,
                      &XCS::set_teletransportation)
        .def_property("GAMMA", &XCS::get_gamma, &XCS::set_gamma)
        .def_property("P_EXPLORE", &XCS::get_p_explore, &XCS::set_p_explore)
        .def_property("EA_SELECT_TYPE", &XCS::get_ea_select_type,
                      &XCS::set_ea_select_type)
        .def_property("EA_SELECT_SIZE", &XCS::get_ea_select_size,
                      &XCS::set_ea_select_size)
        .def_property("THETA_EA", &XCS::get_theta_ea, &XCS::set_theta_ea)
        .def_property("P_CROSSOVER", &XCS::get_p_crossover,
                      &XCS::set_p_crossover)
        .def_property("LAMBDA", &XCS::get_lambda, &XCS::set_lambda)
        .def_property("ERR_REDUC", &XCS::get_err_reduc, &XCS::set_err_reduc)
        .def_property("FIT_REDUC", &XCS::get_fit_reduc, &XCS::set_fit_reduc)
        .def_property("EA_SUBSUMPTION", &XCS::get_ea_subsumption,
                      &XCS::set_ea_subsumption)
        .def_property("EA_PRED_RESET", &XCS::get_ea_pred_reset,
                      &XCS::set_ea_pred_reset)
        .def("time", &XCS::get_time)
        .def("x_dim", &XCS::get_x_dim)
        .def("y_dim", &XCS::get_y_dim)
        .def("n_actions", &XCS::get_n_actions)
        .def("pset_size", &XCS::get_pset_size)
        .def("pset_num", &XCS::get_pset_num)
        .def("pset_mean_cond_size", &XCS::get_pset_mean_cond_size)
        .def("pset_mean_pred_size", &XCS::get_pset_mean_pred_size)
        .def("pset_mean_pred_eta", &XCS::get_pset_mean_pred_eta)
        .def("pset_mean_pred_neurons", &XCS::get_pset_mean_pred_neurons)
        .def("pset_mean_pred_layers", &XCS::get_pset_mean_pred_layers)
        .def("pset_mean_pred_connections", &XCS::get_pset_mean_pred_connections)
        .def("pset_mean_cond_neurons", &XCS::get_pset_mean_cond_neurons)
        .def("pset_mean_cond_layers", &XCS::get_pset_mean_cond_layers)
        .def("pset_mean_cond_connections", &XCS::get_pset_mean_cond_connections)
        .def("mset_size", &XCS::get_mset_size)
        .def("aset_size", &XCS::get_aset_size)
        .def("mfrac", &XCS::get_mfrac)
        .def("print_pset", &XCS::print_pset)
        .def("print_params", &XCS::print_params)
        .def("pred_expand", &XCS::pred_expand)
        .def("ae_to_classifier", &XCS::ae_to_classifier)
        .def("json", &XCS::json_export)
        .def("json_parameters", &XCS::json_parameters);
}
