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
#include "clset.h"
#include "clset_neural.h"
#include "config.h"
#include "pa.h"
#include "param.h"
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
     * @param x_dim The dimensionality of the input variables.
     * @param y_dim The dimensionality of the prediction variables.
     * @param n_actions The total number of possible actions.
     */
    XCS(const int x_dim, const int y_dim, const int n_actions) :
        XCS(x_dim, y_dim, n_actions, "default.ini")
    {
    }

    /**
     * @brief Constructor with a specified config.
     * @param x_dim The dimensionality of the input variables.
     * @param y_dim The dimensionality of the prediction variables.
     * @param n_actions The total number of possible actions.
     * @param filename The name of a parameter configuration file.
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
     * @brief Writes the entire current state of XCSF to a binary file.
     * @param filename String containing the name of the output file.
     * @return The total number of elements written.
     */
    size_t
    save(const char *filename)
    {
        return xcsf_save(&xcs, filename);
    }

    /**
     * @brief Reads the entire current state of XCSF from a binary file.
     * @param filename String containing the name of the input file.
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
     * @param y_dim The output dimension (i.e., the number of classes).
     * @param n_del The number of hidden layers to remove.
     */
    void
    ae_to_classifier(const int y_dim, const int n_del)
    {
        xcsf_ae_to_classifier(&xcs, y_dim, n_del);
    }

    /**
     * @brief Prints the current population.
     * @param print_cond Whether to print the condition.
     * @param print_act Whether to print the action.
     * @param print_pred Whether to print the prediction.
     */
    void
    print_pop(const _Bool print_cond, const _Bool print_act,
              const _Bool print_pred)
    {
        xcsf_print_pop(&xcs, print_cond, print_act, print_pred);
    }

    /* Reinforcement learning */

    /**
     * @brief Creates/updates an action set for a given (state, action, reward).
     * @param input The input state to match.
     * @param action The selected action.
     * @param reward The reward for having performed the action.
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
     * @param input The input state.
     * @param explore Whether this is an exploration step.
     * @return The selected action.
     */
    int
    decision(const py::array_t<double> input, const _Bool explore)
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
     * @param reward The reward from performing the action.
     * @param done Whether the environment is in a terminal state.
     */
    void
    update(const double reward, const _Bool done)
    {
        payoff = reward;
        xcs_rl_update(&xcs, state, action, payoff, done);
    }

    /**
     * @brief Returns the reinforcement learning system prediction error.
     * @param reward The current reward.
     * @param done Whether the environment is in a terminal state.
     * @param max_p The maximum payoff in the environment.
     * @return The prediction error.
     */
    double
    error(const double reward, const _Bool done, const double max_p)
    {
        payoff = reward;
        return xcs_rl_error(&xcs, action, payoff, done, max_p);
    }

    /* Supervised learning */

    /**
     * @brief Executes MAX_TRIALS number of XCSF learning iterations using the
     * provided training data.
     * @param train_X The input values to use for training.
     * @param train_Y The true output values to use for training.
     * @param shuffle Whether to randomise the instances during training.
     * @return The average XCSF training error using the loss function.
     */
    double
    fit(const py::array_t<double> train_X, const py::array_t<double> train_Y,
        const _Bool shuffle)
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
     * @param train_X The input values to use for training.
     * @param train_Y The true output values to use for training.
     * @param test_X The input values to use for testing.
     * @param test_Y The true output values to use for testing.
     * @param shuffle Whether to randomise the instances during training.
     * @return The average XCSF training error using the loss function.
     */
    double
    fit(const py::array_t<double> train_X, const py::array_t<double> train_Y,
        const py::array_t<double> test_X, const py::array_t<double> test_Y,
        const _Bool shuffle)
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
     * @param x The input variables.
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
     * @param test_X The input values to use for scoring.
     * @param test_Y The true output values to use for scoring.
     * @return The average XCSF error using the loss function.
     */
    double
    score(const py::array_t<double> test_X, const py::array_t<double> test_Y)
    {
        return score(test_X, test_Y, 0);
    }

    /**
     * @brief Returns the error using N random samples from the provided data.
     * @param test_X The input values to use for scoring.
     * @param test_Y The true output values to use for scoring.
     * @param N The maximum number of samples to draw randomly for scoring.
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

    py::list
    get_cond_num_neurons(void)
    {
        py::list list;
        for (int i = 0; i < MAX_LAYERS && xcs.COND_NUM_NEURONS[i] > 0; ++i) {
            list.append(xcs.COND_NUM_NEURONS[i]);
        }
        return list;
    }

    py::list
    get_cond_max_neurons(void)
    {
        py::list list;
        for (int i = 0; i < MAX_LAYERS && xcs.COND_MAX_NEURONS[i] > 0; ++i) {
            list.append(xcs.COND_MAX_NEURONS[i]);
        }
        return list;
    }

    py::list
    get_pred_num_neurons(void)
    {
        py::list list;
        for (int i = 0; i < MAX_LAYERS && xcs.PRED_NUM_NEURONS[i] > 0; ++i) {
            list.append(xcs.PRED_NUM_NEURONS[i]);
        }
        return list;
    }

    py::list
    get_pred_max_neurons(void)
    {
        py::list list;
        for (int i = 0; i < MAX_LAYERS && xcs.PRED_MAX_NEURONS[i] > 0; ++i) {
            list.append(xcs.PRED_MAX_NEURONS[i]);
        }
        return list;
    }

    int
    get_omp_num_threads(void)
    {
        return xcs.OMP_NUM_THREADS;
    }

    _Bool
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

    int
    get_act_type(void)
    {
        return xcs.ACT_TYPE;
    }

    int
    get_cond_type(void)
    {
        return xcs.COND_TYPE;
    }

    int
    get_pred_type(void)
    {
        return xcs.PRED_TYPE;
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

    double
    get_max_con(void)
    {
        return xcs.COND_MAX;
    }

    double
    get_min_con(void)
    {
        return xcs.COND_MIN;
    }

    double
    get_cond_smin(void)
    {
        return xcs.COND_SMIN;
    }

    int
    get_cond_bits(void)
    {
        return xcs.COND_BITS;
    }

    _Bool
    get_cond_evolve_weights(void)
    {
        return xcs.COND_EVOLVE_WEIGHTS;
    }

    _Bool
    get_cond_evolve_neurons(void)
    {
        return xcs.COND_EVOLVE_NEURONS;
    }

    _Bool
    get_cond_evolve_functions(void)
    {
        return xcs.COND_EVOLVE_FUNCTIONS;
    }

    _Bool
    get_cond_evolve_connectivity(void)
    {
        return xcs.COND_EVOLVE_CONNECTIVITY;
    }

    int
    get_cond_output_activation(void)
    {
        return xcs.COND_OUTPUT_ACTIVATION;
    }

    int
    get_cond_hidden_activation(void)
    {
        return xcs.COND_HIDDEN_ACTIVATION;
    }

    int
    get_pred_output_activation(void)
    {
        return xcs.PRED_OUTPUT_ACTIVATION;
    }

    int
    get_pred_hidden_activation(void)
    {
        return xcs.PRED_HIDDEN_ACTIVATION;
    }

    double
    get_pred_momentum(void)
    {
        return xcs.PRED_MOMENTUM;
    }

    double
    get_pred_decay(void)
    {
        return xcs.PRED_DECAY;
    }

    _Bool
    get_pred_evolve_weights(void)
    {
        return xcs.PRED_EVOLVE_WEIGHTS;
    }

    _Bool
    get_pred_evolve_neurons(void)
    {
        return xcs.PRED_EVOLVE_NEURONS;
    }

    _Bool
    get_pred_evolve_functions(void)
    {
        return xcs.PRED_EVOLVE_FUNCTIONS;
    }

    _Bool
    get_pred_evolve_connectivity(void)
    {
        return xcs.PRED_EVOLVE_CONNECTIVITY;
    }

    _Bool
    get_pred_evolve_eta(void)
    {
        return xcs.PRED_EVOLVE_ETA;
    }

    _Bool
    get_pred_sgd_weights(void)
    {
        return xcs.PRED_SGD_WEIGHTS;
    }

    _Bool
    get_pred_reset(void)
    {
        return xcs.PRED_RESET;
    }

    int
    get_max_neuron_grow(void)
    {
        return xcs.MAX_NEURON_GROW;
    }

    _Bool
    get_stateful(void)
    {
        return xcs.STATEFUL;
    }

    int
    get_max_k(void)
    {
        return xcs.MAX_K;
    }

    int
    get_max_t(void)
    {
        return xcs.MAX_T;
    }

    int
    get_gp_num_cons(void)
    {
        return xcs.GP_NUM_CONS;
    }

    int
    get_gp_init_depth(void)
    {
        return xcs.GP_INIT_DEPTH;
    }

    double
    get_pred_eta(void)
    {
        return xcs.PRED_ETA;
    }

    double
    get_cond_eta(void)
    {
        return xcs.COND_ETA;
    }

    double
    get_pred_x0(void)
    {
        return xcs.PRED_X0;
    }

    double
    get_pred_rls_scale_factor(void)
    {
        return xcs.PRED_RLS_SCALE_FACTOR;
    }

    double
    get_pred_rls_lambda(void)
    {
        return xcs.PRED_RLS_LAMBDA;
    }

    int
    get_theta_sub(void)
    {
        return xcs.THETA_SUB;
    }

    _Bool
    get_ea_subsumption(void)
    {
        return xcs.EA_SUBSUMPTION;
    }

    _Bool
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

    void
    set_cond_num_neurons(const py::list &a)
    {
        memset(xcs.COND_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
        for (size_t i = 0; i < a.size(); ++i) {
            xcs.COND_NUM_NEURONS[i] = a[i].cast<int>();
        }
    }

    void
    set_cond_max_neurons(const py::list &a)
    {
        memset(xcs.COND_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
        for (size_t i = 0; i < a.size(); ++i) {
            xcs.COND_MAX_NEURONS[i] = a[i].cast<int>();
        }
    }

    void
    set_pred_num_neurons(const py::list &a)
    {
        memset(xcs.PRED_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
        for (size_t i = 0; i < a.size(); ++i) {
            xcs.PRED_NUM_NEURONS[i] = a[i].cast<int>();
        }
    }

    void
    set_pred_max_neurons(const py::list &a)
    {
        memset(xcs.PRED_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
        for (size_t i = 0; i < a.size(); ++i) {
            xcs.PRED_MAX_NEURONS[i] = a[i].cast<int>();
        }
    }

    void
    set_omp_num_threads(const int a)
    {
        param_set_omp_num_threads(&xcs, a);
    }

    void
    set_pop_init(const _Bool a)
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
    set_act_type(const int a)
    {
        param_set_act_type(&xcs, a);
    }

    void
    set_cond_type(const int a)
    {
        param_set_cond_type(&xcs, a);
    }

    void
    set_pred_type(const int a)
    {
        param_set_pred_type(&xcs, a);
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
    set_max_con(const double a)
    {
        param_set_cond_max(&xcs, a);
    }

    void
    set_min_con(const double a)
    {
        param_set_cond_min(&xcs, a);
    }

    void
    set_cond_smin(const double a)
    {
        param_set_cond_smin(&xcs, a);
    }

    void
    set_cond_bits(const int a)
    {
        param_set_cond_bits(&xcs, a);
    }

    void
    set_cond_evolve_weights(const _Bool a)
    {
        param_set_cond_evolve_weights(&xcs, a);
    }

    void
    set_cond_evolve_neurons(const _Bool a)
    {
        param_set_cond_evolve_neurons(&xcs, a);
    }

    void
    set_cond_evolve_functions(const _Bool a)
    {
        param_set_cond_evolve_functions(&xcs, a);
    }

    void
    set_cond_evolve_connectivity(const _Bool a)
    {
        param_set_cond_evolve_connectivity(&xcs, a);
    }

    void
    set_cond_output_activation(const int a)
    {
        param_set_cond_output_activation(&xcs, a);
    }

    void
    set_cond_hidden_activation(const int a)
    {
        param_set_cond_hidden_activation(&xcs, a);
    }

    void
    set_pred_output_activation(const int a)
    {
        param_set_pred_output_activation(&xcs, a);
    }

    void
    set_pred_hidden_activation(const int a)
    {
        param_set_pred_hidden_activation(&xcs, a);
    }

    void
    set_pred_momentum(const double a)
    {
        param_set_pred_momentum(&xcs, a);
    }

    void
    set_pred_decay(const double a)
    {
        param_set_pred_decay(&xcs, a);
    }

    void
    set_pred_evolve_weights(const _Bool a)
    {
        param_set_pred_evolve_weights(&xcs, a);
    }

    void
    set_pred_evolve_neurons(const _Bool a)
    {
        param_set_pred_evolve_neurons(&xcs, a);
    }

    void
    set_pred_evolve_functions(const _Bool a)
    {
        param_set_pred_evolve_functions(&xcs, a);
    }

    void
    set_pred_evolve_connectivity(const _Bool a)
    {
        param_set_pred_evolve_connectivity(&xcs, a);
    }

    void
    set_pred_evolve_eta(const _Bool a)
    {
        param_set_pred_evolve_eta(&xcs, a);
    }

    void
    set_pred_sgd_weights(const _Bool a)
    {
        param_set_pred_sgd_weights(&xcs, a);
    }

    void
    set_pred_reset(const _Bool a)
    {
        param_set_pred_reset(&xcs, a);
    }

    void
    set_max_neuron_grow(const int a)
    {
        param_set_max_neuron_grow(&xcs, a);
    }

    void
    set_stateful(const _Bool a)
    {
        param_set_stateful(&xcs, a);
    }

    void
    set_max_k(const int a)
    {
        param_set_max_k(&xcs, a);
    }

    void
    set_max_t(const int a)
    {
        param_set_max_t(&xcs, a);
    }

    void
    set_gp_num_cons(const int a)
    {
        param_set_gp_num_cons(&xcs, a);
    }

    void
    set_gp_init_depth(const int a)
    {
        param_set_gp_init_depth(&xcs, a);
    }

    void
    set_pred_eta(const double a)
    {
        param_set_pred_eta(&xcs, a);
    }

    void
    set_cond_eta(const double a)
    {
        param_set_cond_eta(&xcs, a);
    }

    void
    set_pred_x0(const double a)
    {
        param_set_pred_x0(&xcs, a);
    }

    void
    set_pred_rls_scale_factor(const double a)
    {
        param_set_pred_rls_scale_factor(&xcs, a);
    }

    void
    set_pred_rls_lambda(const double a)
    {
        param_set_pred_rls_lambda(&xcs, a);
    }

    void
    set_theta_sub(const int a)
    {
        param_set_theta_sub(&xcs, a);
    }

    void
    set_ea_subsumption(const _Bool a)
    {
        param_set_ea_subsumption(&xcs, a);
    }

    void
    set_set_subsumption(const _Bool a)
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
    random_init();

    double (XCS::*fit1)(const py::array_t<double>, const int, const double) =
        &XCS::fit;
    double (XCS::*fit2)(const py::array_t<double>, const py::array_t<double>,
                        const _Bool) = &XCS::fit;
    double (XCS::*fit3)(const py::array_t<double>, const py::array_t<double>,
                        const py::array_t<double>, const py::array_t<double>,
                        const _Bool) = &XCS::fit;
    double (XCS::*score1)(const py::array_t<double> test_X,
                          const py::array_t<double> test_Y) = &XCS::score;
    double (XCS::*score2)(const py::array_t<double> test_X,
                          const py::array_t<double> test_Y, const int N) =
        &XCS::score;

    double (XCS::*error1)(void) = &XCS::error;
    double (XCS::*error2)(const double, const _Bool, const double) =
        &XCS::error;

    py::class_<XCS>(m, "XCS")
        .def(py::init<const int, const int, const int>())
        .def(py::init<const int, const int, const int, const char *>())
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
        .def_property("ACT_TYPE", &XCS::get_act_type, &XCS::set_act_type)
        .def_property("COND_TYPE", &XCS::get_cond_type, &XCS::set_cond_type)
        .def_property("PRED_TYPE", &XCS::get_pred_type, &XCS::set_pred_type)
        .def_property("P_CROSSOVER", &XCS::get_p_crossover,
                      &XCS::set_p_crossover)
        .def_property("THETA_EA", &XCS::get_theta_ea, &XCS::set_theta_ea)
        .def_property("LAMBDA", &XCS::get_lambda, &XCS::set_lambda)
        .def_property("EA_SELECT_TYPE", &XCS::get_ea_select_type,
                      &XCS::set_ea_select_type)
        .def_property("EA_SELECT_SIZE", &XCS::get_ea_select_size,
                      &XCS::set_ea_select_size)
        .def_property("COND_MAX", &XCS::get_max_con, &XCS::set_max_con)
        .def_property("COND_MIN", &XCS::get_min_con, &XCS::set_min_con)
        .def_property("COND_SMIN", &XCS::get_cond_smin, &XCS::set_cond_smin)
        .def_property("COND_BITS", &XCS::get_cond_bits, &XCS::set_cond_bits)
        .def_property("COND_EVOLVE_WEIGHTS", &XCS::get_cond_evolve_weights,
                      &XCS::set_cond_evolve_weights)
        .def_property("COND_EVOLVE_NEURONS", &XCS::get_cond_evolve_neurons,
                      &XCS::set_cond_evolve_neurons)
        .def_property("COND_EVOLVE_FUNCTIONS", &XCS::get_cond_evolve_functions,
                      &XCS::set_cond_evolve_functions)
        .def_property("COND_EVOLVE_CONNECTIVITY",
                      &XCS::get_cond_evolve_connectivity,
                      &XCS::set_cond_evolve_connectivity)
        .def_property("COND_NUM_NEURONS", &XCS::get_cond_num_neurons,
                      &XCS::set_cond_num_neurons)
        .def_property("COND_MAX_NEURONS", &XCS::get_cond_max_neurons,
                      &XCS::set_cond_max_neurons)
        .def_property("COND_OUTPUT_ACTIVATION",
                      &XCS::get_cond_output_activation,
                      &XCS::set_cond_output_activation)
        .def_property("COND_HIDDEN_ACTIVATION",
                      &XCS::get_cond_hidden_activation,
                      &XCS::set_cond_hidden_activation)
        .def_property("PRED_NUM_NEURONS", &XCS::get_pred_num_neurons,
                      &XCS::set_pred_num_neurons)
        .def_property("PRED_MAX_NEURONS", &XCS::get_pred_max_neurons,
                      &XCS::set_pred_max_neurons)
        .def_property("PRED_OUTPUT_ACTIVATION",
                      &XCS::get_pred_output_activation,
                      &XCS::set_pred_output_activation)
        .def_property("PRED_HIDDEN_ACTIVATION",
                      &XCS::get_pred_hidden_activation,
                      &XCS::set_pred_hidden_activation)
        .def_property("PRED_MOMENTUM", &XCS::get_pred_momentum,
                      &XCS::set_pred_momentum)
        .def_property("PRED_DECAY", &XCS::get_pred_decay, &XCS::set_pred_decay)
        .def_property("PRED_EVOLVE_WEIGHTS", &XCS::get_pred_evolve_weights,
                      &XCS::set_pred_evolve_weights)
        .def_property("PRED_EVOLVE_NEURONS", &XCS::get_pred_evolve_neurons,
                      &XCS::set_pred_evolve_neurons)
        .def_property("PRED_EVOLVE_FUNCTIONS", &XCS::get_pred_evolve_functions,
                      &XCS::set_pred_evolve_functions)
        .def_property("PRED_EVOLVE_CONNECTIVITY",
                      &XCS::get_pred_evolve_connectivity,
                      &XCS::set_pred_evolve_connectivity)
        .def_property("PRED_EVOLVE_ETA", &XCS::get_pred_evolve_eta,
                      &XCS::set_pred_evolve_eta)
        .def_property("PRED_SGD_WEIGHTS", &XCS::get_pred_sgd_weights,
                      &XCS::set_pred_sgd_weights)
        .def_property("PRED_RESET", &XCS::get_pred_reset, &XCS::set_pred_reset)
        .def_property("MAX_NEURON_GROW", &XCS::get_max_neuron_grow,
                      &XCS::set_max_neuron_grow)
        .def_property("STATEFUL", &XCS::get_stateful, &XCS::set_stateful)
        .def_property("MAX_K", &XCS::get_max_k, &XCS::set_max_k)
        .def_property("MAX_T", &XCS::get_max_t, &XCS::set_max_t)
        .def_property("GP_NUM_CONS", &XCS::get_gp_num_cons,
                      &XCS::set_gp_num_cons)
        .def_property("GP_INIT_DEPTH", &XCS::get_gp_init_depth,
                      &XCS::set_gp_init_depth)
        .def_property("COND_ETA", &XCS::get_cond_eta, &XCS::set_cond_eta)
        .def_property("PRED_ETA", &XCS::get_pred_eta, &XCS::set_pred_eta)
        .def_property("PRED_X0", &XCS::get_pred_x0, &XCS::set_pred_x0)
        .def_property("PRED_RLS_SCALE_FACTOR", &XCS::get_pred_rls_scale_factor,
                      &XCS::set_pred_rls_scale_factor)
        .def_property("PRED_RLS_LAMBDA", &XCS::get_pred_rls_lambda,
                      &XCS::set_pred_rls_lambda)
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
