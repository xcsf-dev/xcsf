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
 * @copyright The Authors.
 * @date 2020.
 * @brief Python library wrapper functions.
 */

#ifdef _WIN32 // Try to work around https://bugs.python.org/issue11566
    #define _hypot hypot
#endif

#include "../lib/pybind11/include/pybind11/numpy.h"
#include "../lib/pybind11/include/pybind11/pybind11.h"
#include <string>
#include <vector>

namespace py = pybind11;

extern "C" {
#include "action.h"
#include "clset.h"
#include "clset_neural.h"
#include "condition.h"
#include "config.h"
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
     * @brief Constructor with default config.
     * @param [in] x_dim The dimensionality of the input variables.
     * @param [in] y_dim The dimensionality of the prediction variables.
     * @param [in] n_actions The total number of possible actions.
     */
    XCS(const int x_dim, const int y_dim, const int n_actions) :
        XCS(x_dim, y_dim, n_actions, "default.ini")
    {
    }

    /**
     * @brief Constructor with a specified config.
     * @param [in] x_dim The dimensionality of the input variables.
     * @param [in] y_dim The dimensionality of the prediction variables.
     * @param [in] n_actions The total number of possible actions.
     * @param [in] filename The name of a parameter configuration file.
     */
    XCS(const int x_dim, const int y_dim, const int n_actions,
        const char *filename)
    {
        param_init(&xcs);
        config_read(&xcs, filename);
        param_set_x_dim(&xcs, x_dim);
        param_set_y_dim(&xcs, y_dim);
        param_set_n_actions(&xcs, n_actions);
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
        xcsf_store_pop(&xcs);
    }

    /**
     * @brief Retrieves the stored population, setting it as current.
     */
    void
    retrieve(void)
    {
        xcsf_retrieve_pop(&xcs);
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
    print_pop(const bool print_cond, const bool print_act,
              const bool print_pred)
    {
        xcsf_print_pop(&xcs, print_cond, print_act, print_pred);
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
            clset_pop_init(&xcs);
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
        const py::buffer_info buf_x = train_X.request();
        const py::buffer_info buf_y = train_Y.request();
        if (buf_x.shape[0] != buf_y.shape[0]) {
            printf("error: training X and Y n_samples are not equal\n");
            exit(EXIT_FAILURE);
        }
        // load training data
        train_data->n_samples = buf_x.shape[0];
        train_data->x_dim = buf_x.shape[1];
        train_data->y_dim = buf_y.shape[1];
        train_data->x = (double *) buf_x.ptr;
        train_data->y = (double *) buf_y.ptr;
        // first execution
        if (xcs.time == 0) {
            clset_pop_init(&xcs);
        }
        // execute
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
        const py::buffer_info buf_train_x = train_X.request();
        const py::buffer_info buf_train_y = train_Y.request();
        const py::buffer_info buf_test_x = test_X.request();
        const py::buffer_info buf_test_y = test_Y.request();
        if (buf_train_x.shape[0] != buf_train_y.shape[0]) {
            printf("error: training X and Y n_samples are not equal\n");
            exit(EXIT_FAILURE);
        }
        if (buf_test_x.shape[0] != buf_test_y.shape[0]) {
            printf("error: testing X and Y n_samples are not equal\n");
            exit(EXIT_FAILURE);
        }
        if (buf_train_x.shape[1] != buf_test_x.shape[1]) {
            printf("error: number of train and test X cols are not equal\n");
            exit(EXIT_FAILURE);
        }
        if (buf_train_y.shape[1] != buf_test_y.shape[1]) {
            printf("error: number of train and test Y cols are not equal\n");
            exit(EXIT_FAILURE);
        }
        // load training data
        train_data->n_samples = buf_train_x.shape[0];
        train_data->x_dim = buf_train_x.shape[1];
        train_data->y_dim = buf_train_y.shape[1];
        train_data->x = (double *) buf_train_x.ptr;
        train_data->y = (double *) buf_train_y.ptr;
        // load testing data
        test_data->n_samples = buf_test_x.shape[0];
        test_data->x_dim = buf_test_x.shape[1];
        test_data->y_dim = buf_test_y.shape[1];
        test_data->x = (double *) buf_test_x.ptr;
        test_data->y = (double *) buf_test_y.ptr;
        // first execution
        if (xcs.time == 0) {
            clset_pop_init(&xcs);
        }
        // execute
        return xcs_supervised_fit(&xcs, train_data, test_data, shuffle);
    }

    /**
     * @brief Returns the XCSF prediction array for the provided input.
     * @param [in] x The input variables.
     * @return The prediction array values.
     */
    py::array_t<double>
    predict(const py::array_t<double> x)
    {
        // inputs to predict
        const py::buffer_info buf_x = x.request();
        const int n_samples = buf_x.shape[0];
        const double *input = (double *) buf_x.ptr;
        // predicted outputs
        double *output =
            (double *) malloc(sizeof(double) * n_samples * xcs.pa_size);
        xcs_supervised_predict(&xcs, input, output, n_samples);
        // return numpy array
        return py::array_t<double>(
            std::vector<ptrdiff_t>{ n_samples, xcs.pa_size }, output);
    }

    /**
     * @brief Returns the error over one sequential pass of the provided data.
     * @param [in] test_X The input values to use for scoring.
     * @param [in] test_Y The true output values to use for scoring.
     * @return The average XCSF error using the loss function.
     */
    double
    score(const py::array_t<double> test_X, const py::array_t<double> test_Y)
    {
        return score(test_X, test_Y, 0);
    }

    /**
     * @brief Returns the error using N random samples from the provided data.
     * @param [in] test_X The input values to use for scoring.
     * @param [in] test_Y The true output values to use for scoring.
     * @param [in] N The maximum number of samples to draw randomly for scoring.
     * @return The average XCSF error using the loss function.
     */
    double
    score(const py::array_t<double> test_X, const py::array_t<double> test_Y,
          const int N)
    {
        const py::buffer_info buf_x = test_X.request();
        const py::buffer_info buf_y = test_Y.request();
        if (buf_x.shape[0] != buf_y.shape[0]) {
            printf("error: training X and Y n_samples are not equal\n");
            exit(EXIT_FAILURE);
        }
        test_data->n_samples = buf_x.shape[0];
        test_data->x_dim = buf_x.shape[1];
        test_data->y_dim = buf_y.shape[1];
        test_data->x = (double *) buf_x.ptr;
        test_data->y = (double *) buf_y.ptr;
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

    int
    get_loss_func(void)
    {
        return xcs.LOSS_FUNC;
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
    get_eps_0(void)
    {
        return xcs.EPS_0;
    }

    double
    get_err_reduc(void)
    {
        return xcs.ERR_REDUC;
    }

    double
    get_fit_reduc(void)
    {
        return xcs.FIT_REDUC;
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

    int
    get_theta_del(void)
    {
        return xcs.THETA_DEL;
    }

    double
    get_p_crossover(void)
    {
        return xcs.P_CROSSOVER;
    }

    double
    get_theta_ea(void)
    {
        return xcs.THETA_EA;
    }

    int
    get_lambda(void)
    {
        return xcs.LAMBDA;
    }

    int
    get_ea_select_type(void)
    {
        return xcs.EA_SELECT_TYPE;
    }

    double
    get_ea_select_size(void)
    {
        return xcs.EA_SELECT_SIZE;
    }

    int
    get_theta_sub(void)
    {
        return xcs.THETA_SUB;
    }

    bool
    get_ea_subsumption(void)
    {
        return xcs.EA_SUBSUMPTION;
    }

    bool
    get_set_subsumption(void)
    {
        return xcs.SET_SUBSUMPTION;
    }

    int
    get_pop_size(void)
    {
        return xcs.pset.size;
    }

    int
    get_pop_num(void)
    {
        return xcs.pset.num;
    }

    int
    get_time(void)
    {
        return xcs.time;
    }

    double
    get_x_dim(void)
    {
        return xcs.x_dim;
    }

    double
    get_y_dim(void)
    {
        return xcs.y_dim;
    }

    double
    get_n_actions(void)
    {
        return xcs.n_actions;
    }

    double
    get_pop_mean_cond_size(void)
    {
        return clset_mean_cond_size(&xcs, &xcs.pset);
    }

    double
    get_pop_mean_pred_size(void)
    {
        return clset_mean_pred_size(&xcs, &xcs.pset);
    }

    double
    get_pop_mean_pred_eta(const int layer)
    {
        return clset_mean_pred_eta(&xcs, &xcs.pset, layer);
    }

    double
    get_pop_mean_pred_neurons(const int layer)
    {
        return clset_mean_pred_neurons(&xcs, &xcs.pset, layer);
    }

    double
    get_pop_mean_pred_connections(const int layer)
    {
        return clset_mean_pred_connections(&xcs, &xcs.pset, layer);
    }

    double
    get_pop_mean_pred_layers(void)
    {
        return clset_mean_pred_layers(&xcs, &xcs.pset);
    }

    double
    get_pop_mean_cond_connections(const int layer)
    {
        return clset_mean_cond_connections(&xcs, &xcs.pset, layer);
    }

    double
    get_pop_mean_cond_neurons(const int layer)
    {
        return clset_mean_cond_neurons(&xcs, &xcs.pset, layer);
    }

    double
    get_pop_mean_cond_layers(void)
    {
        return clset_mean_cond_layers(&xcs, &xcs.pset);
    }

    double
    get_msetsize(void)
    {
        return xcs.msetsize;
    }

    double
    get_asetsize(void)
    {
        return xcs.asetsize;
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

    /* SETTERS */

    /**
     * @brief Sets the condition type.
     * @param [in] type String representing a name of a condition type.
     */
    void
    set_condition(const std::string type)
    {
        set_condition(type, {});
    }

    /**
     * @brief Sets the action type.
     * @param [in] type String representing a name of an action type.
     */
    void
    set_action(const std::string type)
    {
        set_action(type, {});
    }

    /**
     * @brief Sets the prediction type.
     * @param [in] type String representing a name of a prediction type.
     */
    void
    set_prediction(const std::string type)
    {
        set_prediction(type, {});
    }

    /**
     * @brief Sets the condition type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    set_condition(const std::string type, const py::dict &args)
    {
        if (type == COND_STRING_DUMMY) {
            xcs.COND_TYPE = COND_TYPE_DUMMY;
        } else if (type == COND_STRING_HYPERRECTANGLE) {
            xcs.COND_TYPE = COND_TYPE_HYPERRECTANGLE;
            unpack_cond_csr(args);
        } else if (type == COND_STRING_HYPERELLIPSOID) {
            xcs.COND_TYPE = COND_TYPE_HYPERELLIPSOID;
            unpack_cond_csr(args);
        } else if (type == COND_STRING_NEURAL) {
            xcs.COND_TYPE = COND_TYPE_NEURAL;
            unpack_cond_neural(args);
        } else if (type == COND_STRING_GP) {
            xcs.COND_TYPE = COND_TYPE_GP;
            unpack_cond_gp(args);
        } else if (type == COND_STRING_DGP) {
            xcs.COND_TYPE = COND_TYPE_DGP;
            unpack_cond_dgp(args);
        } else if (type == COND_STRING_TERNARY) {
            xcs.COND_TYPE = COND_TYPE_TERNARY;
            unpack_cond_ternary(args);
        } else if (type == COND_STRING_RULE_DGP) {
            xcs.COND_TYPE = RULE_TYPE_DGP;
            unpack_cond_dgp(args);
        } else if (type == COND_STRING_RULE_NEURAL) {
            xcs.COND_TYPE = RULE_TYPE_NEURAL;
            unpack_cond_neural(args);
        } else if (type == COND_STRING_RULE_NETWORK) {
            xcs.COND_TYPE = RULE_TYPE_NETWORK;
            unpack_cond_neural(args);
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
                const auto value = item.second.cast<double>();
                param_set_cond_min(&xcs, value);
            } else if (name == "max") {
                const auto value = item.second.cast<double>();
                param_set_cond_max(&xcs, value);
            } else if (name == "spread-min") {
                const auto value = item.second.cast<double>();
                param_set_cond_smin(&xcs, value);
            } else if (name == "eta") {
                const auto value = item.second.cast<double>();
                param_set_cond_eta(&xcs, value);
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
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "num-cons") {
                const auto value = item.second.cast<int>();
                param_set_gp_num_cons(&xcs, value);
            } else if (name == "init-depth") {
                const auto value = item.second.cast<int>();
                param_set_gp_init_depth(&xcs, value);
            }
        }
    }

    /**
     * @brief Sets parameters used by dynamical GP graph conditions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_cond_dgp(const py::dict &args)
    {
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "max-k") {
                const auto value = item.second.cast<int>();
                param_set_max_k(&xcs, value);
            } else if (name == "max-t") {
                const auto value = item.second.cast<int>();
                param_set_max_t(&xcs, value);
            } else if (name == "max-neuron-grow") {
                const auto value = item.second.cast<int>();
                param_set_max_neuron_grow(&xcs, value);
            } else if (name == "stateful") {
                const auto value = item.second.cast<bool>();
                param_set_stateful(&xcs, value);
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
                const auto value = item.second.cast<int>();
                param_set_cond_bits(&xcs, value);
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
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "max-neuron-grow") {
                const auto value = item.second.cast<int>();
                param_set_max_neuron_grow(&xcs, value);
            } else if (name == "evolve-weights") {
                const auto value = item.second.cast<bool>();
                param_set_cond_evolve_weights(&xcs, value);
            } else if (name == "evolve-neurons") {
                const auto value = item.second.cast<bool>();
                param_set_cond_evolve_neurons(&xcs, value);
            } else if (name == "evolve-functions") {
                const auto value = item.second.cast<bool>();
                param_set_cond_evolve_functions(&xcs, value);
            } else if (name == "evolve-connectivity") {
                const auto value = item.second.cast<bool>();
                param_set_cond_evolve_connectivity(&xcs, value);
            } else if (name == "output-activation") {
                const auto value = item.second.cast<std::string>();
                param_set_cond_output_activation_string(&xcs, value.c_str());
            } else if (name == "hidden-activation") {
                const auto value = item.second.cast<std::string>();
                param_set_cond_hidden_activation_string(&xcs, value.c_str());
            } else if (name == "num-neurons") {
                const auto value = item.second.cast<py::list>();
                memset(xcs.COND_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
                for (size_t i = 0; i < value.size(); ++i) {
                    xcs.COND_NUM_NEURONS[i] = value[i].cast<int>();
                }
            } else if (name == "max-neurons") {
                const auto value = item.second.cast<py::list>();
                memset(xcs.COND_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
                for (size_t i = 0; i < value.size(); ++i) {
                    xcs.COND_MAX_NEURONS[i] = value[i].cast<int>();
                }
            }
        }
    }

    /**
     * @brief Sets the action type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    set_action(const std::string type, const py::dict &args)
    {
        if (type == ACT_STRING_INTEGER) {
            xcs.ACT_TYPE = ACT_TYPE_INTEGER;
        } else if (type == ACT_STRING_NEURAL) {
            xcs.ACT_TYPE = ACT_TYPE_NEURAL;
            unpack_cond_neural(args);
        }
    }

    /**
     * @brief Sets the prediction type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    set_prediction(const std::string type, const py::dict &args)
    {
        if (type == PRED_STRING_CONSTANT) {
            xcs.PRED_TYPE = PRED_TYPE_CONSTANT;
        } else if (type == PRED_STRING_NLMS_LINEAR) {
            xcs.PRED_TYPE = PRED_TYPE_NLMS_LINEAR;
            unpack_pred_nlms(args);
        } else if (type == PRED_STRING_NLMS_QUADRATIC) {
            xcs.PRED_TYPE = PRED_TYPE_NLMS_QUADRATIC;
            unpack_pred_nlms(args);
        } else if (type == PRED_STRING_RLS_LINEAR) {
            xcs.PRED_TYPE = PRED_TYPE_RLS_LINEAR;
            unpack_pred_rls(args);
        } else if (type == PRED_STRING_RLS_QUADRATIC) {
            xcs.PRED_TYPE = PRED_TYPE_RLS_QUADRATIC;
            unpack_pred_rls(args);
        } else if (type == PRED_STRING_NEURAL) {
            xcs.PRED_TYPE = PRED_TYPE_NEURAL;
            unpack_pred_neural(args);
        }
    }

    /**
     * @brief Sets parameters used by neural network predictions.
     * @param [in] args Python dictionary of argument name:value pairs.
     */
    void
    unpack_pred_neural(const py::dict &args)
    {
        for (std::pair<py::handle, py::handle> item : args) {
            auto name = item.first.cast<std::string>();
            if (name == "max-neuron-grow") {
                const auto value = item.second.cast<int>();
                param_set_max_neuron_grow(&xcs, value);
            } else if (name == "reset") {
                const auto value = item.second.cast<bool>();
                param_set_pred_reset(&xcs, value);
            } else if (name == "eta") {
                const auto value = item.second.cast<bool>();
                param_set_pred_eta(&xcs, value);
            } else if (name == "evolve-eta") {
                const auto value = item.second.cast<bool>();
                param_set_pred_evolve_eta(&xcs, value);
            } else if (name == "evolve-weights") {
                const auto value = item.second.cast<bool>();
                param_set_pred_evolve_weights(&xcs, value);
            } else if (name == "evolve-neurons") {
                const auto value = item.second.cast<bool>();
                param_set_pred_evolve_neurons(&xcs, value);
            } else if (name == "evolve-functions") {
                const auto value = item.second.cast<bool>();
                param_set_pred_evolve_functions(&xcs, value);
            } else if (name == "evolve-connectivity") {
                const auto value = item.second.cast<bool>();
                param_set_pred_evolve_connectivity(&xcs, value);
            } else if (name == "sgd-weights") {
                const auto value = item.second.cast<bool>();
                param_set_pred_sgd_weights(&xcs, value);
            } else if (name == "momentum") {
                const auto value = item.second.cast<double>();
                param_set_pred_momentum(&xcs, value);
            } else if (name == "decay") {
                const auto value = item.second.cast<double>();
                param_set_pred_decay(&xcs, value);
            } else if (name == "output-activation") {
                const auto value = item.second.cast<std::string>();
                param_set_pred_output_activation_string(&xcs, value.c_str());
            } else if (name == "hidden-activation") {
                const auto value = item.second.cast<std::string>();
                param_set_pred_hidden_activation_string(&xcs, value.c_str());
            } else if (name == "num-neurons") {
                const auto value = item.second.cast<py::list>();
                memset(xcs.PRED_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
                for (size_t i = 0; i < value.size(); ++i) {
                    xcs.PRED_NUM_NEURONS[i] = value[i].cast<int>();
                }
            } else if (name == "max-neurons") {
                const auto value = item.second.cast<py::list>();
                memset(xcs.PRED_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
                for (size_t i = 0; i < value.size(); ++i) {
                    xcs.PRED_MAX_NEURONS[i] = value[i].cast<int>();
                }
            }
        }
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
                const auto value = item.second.cast<double>();
                param_set_pred_x0(&xcs, value);
            } else if (name == "eta") {
                const auto value = item.second.cast<double>();
                param_set_pred_eta(&xcs, value);
            } else if (name == "evolve_eta") {
                const auto value = item.second.cast<bool>();
                param_set_pred_evolve_eta(&xcs, value);
            } else if (name == "reset") {
                const auto value = item.second.cast<bool>();
                param_set_pred_reset(&xcs, value);
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
                const auto value = item.second.cast<double>();
                param_set_pred_x0(&xcs, value);
            } else if (name == "reset") {
                const auto value = item.second.cast<bool>();
                param_set_pred_reset(&xcs, value);
            } else if (name == "rls-scale-factor") {
                const auto value = item.second.cast<double>();
                param_set_pred_rls_scale_factor(&xcs, value);
            } else if (name == "rls-lambda") {
                const auto value = item.second.cast<double>();
                param_set_pred_rls_lambda(&xcs, value);
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
    set_loss_func(const int a)
    {
        param_set_loss_func(&xcs, a);
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
    set_eps_0(const double a)
    {
        param_set_eps_0(&xcs, a);
    }

    void
    set_err_reduc(const double a)
    {
        param_set_err_reduc(&xcs, a);
    }

    void
    set_fit_reduc(const double a)
    {
        param_set_fit_reduc(&xcs, a);
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
    set_p_crossover(const double a)
    {
        param_set_p_crossover(&xcs, a);
    }

    void
    set_theta_ea(const double a)
    {
        param_set_theta_ea(&xcs, a);
    }

    void
    set_lambda(const int a)
    {
        param_set_lambda(&xcs, a);
    }

    void
    set_ea_select_type(const int a)
    {
        param_set_ea_select_type(&xcs, a);
    }

    void
    set_ea_select_size(const double a)
    {
        param_set_ea_select_size(&xcs, a);
    }

    void
    set_theta_sub(const int a)
    {
        param_set_theta_sub(&xcs, a);
    }

    void
    set_ea_subsumption(const bool a)
    {
        param_set_ea_subsumption(&xcs, a);
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
    set_gamma(const double a)
    {
        param_set_gamma(&xcs, a);
    }

    void
    set_p_explore(const double a)
    {
        param_set_p_explore(&xcs, a);
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

    void (XCS::*condition1)(const std::string) = &XCS::set_condition;
    void (XCS::*condition2)(const std::string, const py::dict &) =
        &XCS::set_condition;

    void (XCS::*action1)(const std::string) = &XCS::set_action;
    void (XCS::*action2)(const std::string, const py::dict &) =
        &XCS::set_action;

    void (XCS::*prediction1)(const std::string) = &XCS::set_prediction;
    void (XCS::*prediction2)(const std::string, const py::dict &) =
        &XCS::set_prediction;

    py::class_<XCS>(m, "XCS")
        .def(py::init<const int, const int, const int>())
        .def(py::init<const int, const int, const int, const char *>())
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
        .def_property("OMP_NUM_THREADS", &XCS::get_omp_num_threads,
                      &XCS::set_omp_num_threads)
        .def_property("POP_INIT", &XCS::get_pop_init, &XCS::set_pop_init)
        .def_property("MAX_TRIALS", &XCS::get_max_trials, &XCS::set_max_trials)
        .def_property("PERF_TRIALS", &XCS::get_perf_trials,
                      &XCS::set_perf_trials)
        .def_property("POP_SIZE", &XCS::get_pop_max_size,
                      &XCS::set_pop_max_size)
        .def_property("LOSS_FUNC", &XCS::get_loss_func, &XCS::set_loss_func)
        .def_property("ALPHA", &XCS::get_alpha, &XCS::set_alpha)
        .def_property("BETA", &XCS::get_beta, &XCS::set_beta)
        .def_property("DELTA", &XCS::get_delta, &XCS::set_delta)
        .def_property("EPS_0", &XCS::get_eps_0, &XCS::set_eps_0)
        .def_property("ERR_REDUC", &XCS::get_err_reduc, &XCS::set_err_reduc)
        .def_property("FIT_REDUC", &XCS::get_fit_reduc, &XCS::set_fit_reduc)
        .def_property("INIT_ERROR", &XCS::get_init_error, &XCS::set_init_error)
        .def_property("INIT_FITNESS", &XCS::get_init_fitness,
                      &XCS::set_init_fitness)
        .def_property("NU", &XCS::get_nu, &XCS::set_nu)
        .def_property("M_PROBATION", &XCS::get_m_probation,
                      &XCS::set_m_probation)
        .def_property("THETA_DEL", &XCS::get_theta_del, &XCS::set_theta_del)
        .def_property("P_CROSSOVER", &XCS::get_p_crossover,
                      &XCS::set_p_crossover)
        .def_property("THETA_EA", &XCS::get_theta_ea, &XCS::set_theta_ea)
        .def_property("LAMBDA", &XCS::get_lambda, &XCS::set_lambda)
        .def_property("EA_SELECT_TYPE", &XCS::get_ea_select_type,
                      &XCS::set_ea_select_type)
        .def_property("EA_SELECT_SIZE", &XCS::get_ea_select_size,
                      &XCS::set_ea_select_size)
        .def_property("THETA_SUB", &XCS::get_theta_sub, &XCS::set_theta_sub)
        .def_property("EA_SUBSUMPTION", &XCS::get_ea_subsumption,
                      &XCS::set_ea_subsumption)
        .def_property("SET_SUBSUMPTION", &XCS::get_set_subsumption,
                      &XCS::set_set_subsumption)
        .def_property("TELETRANSPORTATION", &XCS::get_teletransportation,
                      &XCS::set_teletransportation)
        .def_property("GAMMA", &XCS::get_gamma, &XCS::set_gamma)
        .def_property("P_EXPLORE", &XCS::get_p_explore, &XCS::set_p_explore)
        .def("pop_size", &XCS::get_pop_size)
        .def("pop_num", &XCS::get_pop_num)
        .def("time", &XCS::get_time)
        .def("x_dim", &XCS::get_x_dim)
        .def("y_dim", &XCS::get_y_dim)
        .def("n_actions", &XCS::get_n_actions)
        .def("pop_mean_cond_size", &XCS::get_pop_mean_cond_size)
        .def("pop_mean_pred_size", &XCS::get_pop_mean_pred_size)
        .def("pop_mean_pred_eta", &XCS::get_pop_mean_pred_eta)
        .def("pop_mean_pred_neurons", &XCS::get_pop_mean_pred_neurons)
        .def("pop_mean_pred_layers", &XCS::get_pop_mean_pred_layers)
        .def("pop_mean_pred_connections", &XCS::get_pop_mean_pred_connections)
        .def("pop_mean_cond_neurons", &XCS::get_pop_mean_cond_neurons)
        .def("pop_mean_cond_layers", &XCS::get_pop_mean_cond_layers)
        .def("pop_mean_cond_connections", &XCS::get_pop_mean_cond_connections)
        .def("print_pop", &XCS::print_pop)
        .def("msetsize", &XCS::get_msetsize)
        .def("asetsize", &XCS::get_asetsize)
        .def("mfrac", &XCS::get_mfrac)
        .def("print_params", &XCS::print_params)
        .def("pred_expand", &XCS::pred_expand)
        .def("ae_to_classifier", &XCS::ae_to_classifier);
}
