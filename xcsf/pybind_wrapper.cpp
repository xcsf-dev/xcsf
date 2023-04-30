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
 * @date 2020--2023.
 * @brief Python library wrapper functions.
 */

#ifdef _WIN32 // Try to work around https://bugs.python.org/issue11566
    #define _hypot hypot
#endif

#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

extern "C" {
#include "action.h"
#include "clset.h"
#include "clset_neural.h"
#include "condition.h"
#include "ea.h"
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
        catch_error(param_set_n_actions(&xcs, n_actions));
        catch_error(param_set_x_dim(&xcs, x_dim));
        catch_error(param_set_y_dim(&xcs, y_dim));
        xcsf_init(&xcs);
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
     * @param [in] condition Whether to print the condition.
     * @param [in] action Whether to print the action.
     * @param [in] prediction Whether to print the prediction.
     */
    void
    print_pset(const bool condition, const bool action, const bool prediction)
    {
        xcsf_print_pset(&xcs, condition, action, prediction);
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
            std::ostringstream error;
            error << "fit(): x_dim is not equal to: " << xcs.x_dim << std::endl;
            throw std::invalid_argument(error.str());
        }
        if (action < 0 || action >= xcs.n_actions) {
            std::ostringstream error;
            error << "fit(): action outside: [0," << xcs.n_actions << ")"
                  << std::endl;
            throw std::invalid_argument(error.str());
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
            std::ostringstream error;
            error << "decision(): x_dim is not equal to: " << xcs.x_dim;
            throw std::invalid_argument(error.str());
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
            std::string error =
                "load_input(): X and Y n_samples are not equal.";
            throw std::invalid_argument(error);
        }
        if (buf_x.shape[1] != xcs.x_dim) {
            std::ostringstream error;
            error << "load_input(): x_dim != " << xcs.x_dim << std::endl;
            error << "2-D arrays are required. Perhaps reshape your data.";
            throw std::invalid_argument(error.str());
        }
        if (buf_y.shape[1] != xcs.y_dim) {
            std::ostringstream error;
            error << "load_input(): y_dim != " << xcs.y_dim << std::endl;
            error << "2-D arrays are required. Perhaps reshape your data.";
            throw std::invalid_argument(error.str());
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
        return xcs_supervised_fit(&xcs, train_data, test_data, shuffle);
    }

    /**
     * @brief Returns the values specified in the cover array.
     * @param [in] cover The values to return for covering.
     * @return The cover array values.
     */
    double *
    get_cover(const py::array_t<double> cover)
    {
        const py::buffer_info buf_c = cover.request();
        if (buf_c.ndim != 1) {
            std::ostringstream err;
            err << "cover must be an array of shape (1, " << xcs.y_dim << ")"
                << std::endl;
            throw std::invalid_argument(err.str());
        }
        if (buf_c.shape[0] != xcs.y_dim) {
            std::ostringstream err;
            err << "cover length = " << buf_c.shape[0] << " but expected "
                << xcs.y_dim << std::endl;
            throw std::invalid_argument(err.str());
        }
        return reinterpret_cast<double *>(buf_c.ptr);
    }

    /**
     * @brief Returns the XCSF prediction array for the provided input.
     * @param [in] X The input variables.
     * @param [in] cover If cover is not NULL and the match set is empty, the
     * prediction array will be set to this value instead of covering.
     * @return The prediction array values.
     */
    py::array_t<double>
    get_predictions(const py::array_t<double> X, const double *cover)
    {
        const py::buffer_info buf_x = X.request();
        const int n_samples = buf_x.shape[0];
        if (buf_x.shape[1] != xcs.x_dim) {
            std::ostringstream err;
            err << "predict(): x_dim (" << buf_x.shape[1]
                << ") is not equal to: " << xcs.x_dim << std::endl;
            err << "2-D arrays are required. Perhaps reshape your data.";
            throw std::invalid_argument(err.str());
        }
        const double *input = reinterpret_cast<double *>(buf_x.ptr);
        double *output =
            (double *) malloc(sizeof(double) * n_samples * xcs.pa_size);
        xcs_supervised_predict(&xcs, input, output, n_samples, cover);
        return py::array_t<double>(
            std::vector<ptrdiff_t>{ n_samples, xcs.pa_size }, output);
    }

    /**
     * @brief Returns the XCSF prediction array for the provided input.
     * @param [in] X The input variables.
     * @param [in] cover If the match set is empty, the prediction array will
     * be set to this value instead of covering.
     * @return The prediction array values.
     */
    py::array_t<double>
    predict(const py::array_t<double> X, const py::array_t<double> cover)
    {
        const double *cov = get_cover(cover);
        return get_predictions(X, cov);
    }

    /**
     * @brief Returns the XCSF prediction array for the provided input,
     * and executes covering for samples where the match set is empty.
     * @param [in] X The input variables.
     * @return The prediction array values.
     */
    py::array_t<double>
    predict(const py::array_t<double> X)
    {
        return get_predictions(X, NULL);
    }

    /**
     * @brief Returns the error using N random samples from the provided data.
     * @param [in] X The input values to use for scoring.
     * @param [in] Y The true output values to use for scoring.
     * @param [in] N The maximum number of samples to draw randomly for scoring.
     * @param [in] cover If the match set is empty, the prediction array will
     * be set to this value instead of covering.
     * @return The average XCSF error using the loss function.
     */
    double
    get_score(const py::array_t<double> X, const py::array_t<double> Y,
              const int N, const double *cover)
    {
        load_input(test_data, X, Y);
        if (N > 1) {
            return xcs_supervised_score_n(&xcs, test_data, N, cover);
        }
        return xcs_supervised_score(&xcs, test_data, cover);
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
        return get_score(X, Y, N, NULL);
    }

    /**
     * @brief Returns the error using N random samples from the provided data.
     * @param [in] X The input values to use for scoring.
     * @param [in] Y The true output values to use for scoring.
     * @param [in] N The maximum number of samples to draw randomly for scoring.
     * @param [in] cover If the match set is empty, the prediction array will
     * be set to this value instead of covering.
     * @return The average XCSF error using the loss function.
     */
    double
    score(const py::array_t<double> X, const py::array_t<double> Y, const int N,
          const py::array_t<double> cover)
    {
        const double *cov = get_cover(cover);
        return get_score(X, Y, N, cov);
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
        if (cond_param_set_type_string(&xcs, type.c_str()) ==
            COND_TYPE_INVALID) {
            std::ostringstream msg;
            msg << "Invalid condition type: " << type << ". Options: {"
                << COND_TYPE_OPTIONS << "}" << std::endl;
            throw std::invalid_argument(msg.str());
        }
    }

    /**
     * @brief Sets the action type.
     * @param [in] type String representing a name of an action type.
     */
    void
    set_action(const std::string &type)
    {
        if (action_param_set_type_string(&xcs, type.c_str()) ==
            ACT_TYPE_INVALID) {
            std::ostringstream msg;
            msg << "Invalid action type: " << type << ". Options: {"
                << ACT_TYPE_OPTIONS << "}" << std::endl;
            throw std::invalid_argument(msg.str());
        }
    }

    /**
     * @brief Sets the prediction type.
     * @param [in] type String representing a name of a prediction type.
     */
    void
    set_prediction(const std::string &type)
    {
        if (pred_param_set_type_string(&xcs, type.c_str()) ==
            PRED_TYPE_INVALID) {
            std::ostringstream msg;
            msg << "Invalid prediction type: " << type << ". Options: {"
                << PRED_TYPE_OPTIONS << "}" << std::endl;
            throw std::invalid_argument(msg.str());
        }
    }

    /**
     * @brief Finds and replaces all occurrences of a substring.
     * @param [in] subject String to be searched and replaced.
     * @param [in] search String within the subject to find.
     * @param [in] replace String to replace the found text with.
     * @return String.
     */
    std::string
    find_replace_all(std::string subject, const std::string &search,
                     const std::string &replace)
    {
        size_t pos = 0;
        while ((pos = subject.find(search, pos)) != std::string::npos) {
            subject.replace(pos, search.length(), replace);
            pos += replace.length();
        }
        return subject;
    }

    /**
     * @brief Converts a Python dictionary to a C++ string (JSON).
     * @param [in] kwargs Python dictionary of argument name:value pairs.
     * @return JSON string representation of a dictionary.
     */
    cJSON *
    dict_to_json(const py::dict &kwargs)
    {
        py::str s = py::str(*kwargs);
        std::string cs = s.cast<std::string>();
        cs = find_replace_all(cs, "True", "true");
        cs = find_replace_all(cs, "False", "false");
        cs = find_replace_all(cs, "\'", "\"");
        cJSON *json = cJSON_Parse(cs.c_str());
        utils_json_parse_check(json);
        return json;
    }

    /**
     * @brief Sets the condition type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] kwargs Python dictionary of argument name:value pairs.
     */
    void
    set_condition(const std::string &type, const py::dict &kwargs)
    {
        set_condition(type);
        cJSON *args = dict_to_json(kwargs);
        const char *ret = cond_param_json_import(&xcs, args);
        if (ret != NULL) {
            std::ostringstream msg;
            msg << "Invalid condition parameter: " << ret << std::endl;
            throw std::invalid_argument(msg.str());
        }
        cJSON_Delete(args);
    }

    /**
     * @brief Sets the action type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] kwargs Python dictionary of argument name:value pairs.
     */
    void
    set_action(const std::string &type, const py::dict &kwargs)
    {
        set_action(type);
        cJSON *args = dict_to_json(kwargs);
        const char *ret = action_param_json_import(&xcs, args);
        if (ret != NULL) {
            std::ostringstream msg;
            msg << "Invalid action parameter: " << ret << std::endl;
            throw std::invalid_argument(msg.str());
        }
        cJSON_Delete(args);
    }

    /**
     * @brief Sets the prediction type and initialisation arguments.
     * @param [in] type String representing a name of a condition type.
     * @param [in] kwargs Python dictionary of argument name:value pairs.
     */
    void
    set_prediction(const std::string &type, const py::dict &kwargs)
    {
        set_prediction(type);
        cJSON *args = dict_to_json(kwargs);
        const char *ret = pred_param_json_import(&xcs, args);
        if (ret != NULL) {
            std::ostringstream msg;
            msg << "Invalid prediction parameter: " << ret << std::endl;
            throw std::invalid_argument(msg.str());
        }
        cJSON_Delete(args);
    }

    void
    catch_error(const char *ret)
    {
        if (ret != NULL) {
            std::ostringstream msg;
            msg << ret << std::endl;
            throw std::invalid_argument(msg.str());
        }
    }

    void
    set_omp_num_threads(const int a)
    {
        catch_error(param_set_omp_num_threads(&xcs, a));
    }

    void
    set_pop_init(const bool a)
    {
        catch_error(param_set_pop_init(&xcs, a));
    }

    void
    set_max_trials(const int a)
    {
        catch_error(param_set_max_trials(&xcs, a));
    }

    void
    set_perf_trials(const int a)
    {
        catch_error(param_set_perf_trials(&xcs, a));
    }

    void
    set_pop_max_size(const int a)
    {
        catch_error(param_set_pop_size(&xcs, a));
    }

    void
    set_loss_func(const char *a)
    {
        if (param_set_loss_func_string(&xcs, a) == PARAM_INVALID) {
            std::ostringstream msg;
            msg << "Invalid loss function: " << a << ". Options: {"
                << LOSS_OPTIONS << "}" << std::endl;
            throw std::invalid_argument(msg.str());
        }
    }

    void
    set_huber_delta(const double a)
    {
        catch_error(param_set_huber_delta(&xcs, a));
    }

    void
    set_alpha(const double a)
    {
        catch_error(param_set_alpha(&xcs, a));
    }

    void
    set_beta(const double a)
    {
        catch_error(param_set_beta(&xcs, a));
    }

    void
    set_delta(const double a)
    {
        catch_error(param_set_delta(&xcs, a));
    }

    void
    set_e0(const double a)
    {
        catch_error(param_set_e0(&xcs, a));
    }

    void
    set_init_error(const double a)
    {
        catch_error(param_set_init_error(&xcs, a));
    }

    void
    set_init_fitness(const double a)
    {
        catch_error(param_set_init_fitness(&xcs, a));
    }

    void
    set_nu(const double a)
    {
        catch_error(param_set_nu(&xcs, a));
    }

    void
    set_m_probation(const int a)
    {
        catch_error(param_set_m_probation(&xcs, a));
    }

    void
    set_theta_del(const int a)
    {
        catch_error(param_set_theta_del(&xcs, a));
    }

    void
    set_theta_sub(const int a)
    {
        catch_error(param_set_theta_sub(&xcs, a));
    }

    void
    set_set_subsumption(const bool a)
    {
        catch_error(param_set_set_subsumption(&xcs, a));
    }

    void
    set_teletransportation(const int a)
    {
        catch_error(param_set_teletransportation(&xcs, a));
    }

    void
    set_stateful(const bool a)
    {
        catch_error(param_set_stateful(&xcs, a));
    }

    void
    set_compaction(const bool a)
    {
        catch_error(param_set_compaction(&xcs, a));
    }

    void
    set_gamma(const double a)
    {
        catch_error(param_set_gamma(&xcs, a));
    }

    void
    set_p_explore(const double a)
    {
        catch_error(param_set_p_explore(&xcs, a));
    }

    void
    set_ea_select_type(const char *a)
    {
        if (ea_param_set_type_string(&xcs, a) == EA_SELECT_INVALID) {
            std::ostringstream msg;
            msg << "Invalid EA SELECT_TYPE: " << a << ". Options: {"
                << EA_SELECT_OPTIONS << "}" << std::endl;
            throw std::invalid_argument(msg.str());
        }
    }

    void
    set_ea_select_size(const double a)
    {
        catch_error(ea_param_set_select_size(&xcs, a));
    }

    void
    set_theta_ea(const double a)
    {
        catch_error(ea_param_set_theta(&xcs, a));
    }

    void
    set_lambda(const int a)
    {
        catch_error(ea_param_set_lambda(&xcs, a));
    }

    void
    set_p_crossover(const double a)
    {
        catch_error(ea_param_set_p_crossover(&xcs, a));
    }

    void
    set_err_reduc(const double a)
    {
        catch_error(ea_param_set_err_reduc(&xcs, a));
    }

    void
    set_fit_reduc(const double a)
    {
        catch_error(ea_param_set_fit_reduc(&xcs, a));
    }

    void
    set_ea_subsumption(const bool a)
    {
        catch_error(ea_param_set_subsumption(&xcs, a));
    }

    void
    set_ea_pred_reset(const bool a)
    {
        catch_error(ea_param_set_pred_reset(&xcs, a));
    }

    void
    seed(const uint32_t seed)
    {
        rand_init_seed(seed);
    }

    /* JSON */

    /**
     * @brief Returns a JSON formatted string representing the population set.
     * @param [in] condition Whether to return the condition.
     * @param [in] action Whether to return the action.
     * @param [in] prediction Whether to return the prediction.
     * @return String encoded in json format.
     */
    const char *
    json_export(const bool condition, const bool action, const bool prediction)
    {
        if (xcs.pset.list != NULL) {
            return clset_json_export(&xcs, &xcs.pset, condition, action,
                                     prediction);
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

    /**
     * @brief Creates a classifier from JSON and inserts into the population.
     * @param [in] json_str JSON formatted string representing a classifier.
     */
    void
    json_insert_cl(const std::string &json_str)
    {
        cJSON *json = cJSON_Parse(json_str.c_str());
        utils_json_parse_check(json);
        clset_json_insert(&xcs, json);
        cJSON_Delete(json);
    }

    /**
     * @brief Creates classifiers from JSON and inserts into the population.
     * @param [in] json_str JSON formatted string representing a classifier.
     */
    void
    json_insert(const std::string &json_str)
    {
        cJSON *json = cJSON_Parse(json_str.c_str());
        utils_json_parse_check(json);
        if (json->child != NULL && cJSON_IsArray(json->child)) {
            cJSON *tail = json->child->child; // insert inverted for consistency
            tail->prev = NULL; // this should have been set by cJSON!
            while (tail->next != NULL) {
                tail = tail->next;
            }
            while (tail != NULL) {
                clset_json_insert(&xcs, tail);
                tail = tail->prev;
            }
        }
        cJSON_Delete(json);
    }

    /**
     * @brief Writes the current population set to a file in JSON.
     * @param [in] filename Name of the output file.
     */
    void
    json_write(const std::string &filename)
    {
        std::ofstream outfile(filename);
        outfile << json_export(true, true, true);
        outfile.close();
    }

    /**
     * @brief Reads classifiers from a JSON file and adds to the population.
     * @param [in] filename Name of the input file.
     */
    void
    json_read(const std::string &filename)
    {
        std::ifstream infile(filename);
        std::stringstream buffer;
        buffer << infile.rdbuf();
        json_insert(buffer.str());
    }
};

PYBIND11_MODULE(xcsf, m)
{
    m.doc() = "XCSF learning classifier: rule-based online evolutionary "
              "machine learning.\nFor details on how to use this module see: "
              "https://github.com/rpreen/xcsf/wiki/Python-Library-Usage";
    rand_init();

    double (XCS::*fit1)(const py::array_t<double>, const int, const double) =
        &XCS::fit;
    double (XCS::*fit2)(const py::array_t<double>, const py::array_t<double>,
                        const bool) = &XCS::fit;
    double (XCS::*fit3)(const py::array_t<double>, const py::array_t<double>,
                        const py::array_t<double>, const py::array_t<double>,
                        const bool) = &XCS::fit;

    py::array_t<double> (XCS::*predict1)(const py::array_t<double> test_X) =
        &XCS::predict;
    py::array_t<double> (XCS::*predict2)(const py::array_t<double> test_X,
                                         const py::array_t<double> cover) =
        &XCS::predict;

    double (XCS::*score1)(const py::array_t<double> X,
                          const py::array_t<double> Y, const int N) =
        &XCS::score;

    double (XCS::*score2)(const py::array_t<double> X,
                          const py::array_t<double> Y, const int N,
                          const py::array_t<double> cover) = &XCS::score;

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
        .def(py::init<const int, const int, const int>(),
             "Creates a new XCSF class.", py::arg("x_dim"), py::arg("y_dim"),
             py::arg("n_actions"))
        .def("condition", condition1,
             "Sets the condition type. Options: {" COND_TYPE_OPTIONS "}.",
             py::arg("type"))
        .def("condition", condition2,
             "Sets the condition type and args. Options: {" COND_TYPE_OPTIONS
             "}.",
             py::arg("type"), py::arg("args"))
        .def("action", action1,
             "Sets the action type. Options: {" ACT_TYPE_OPTIONS "}.",
             py::arg("type"))
        .def("action", action2,
             "Sets the action type and args. Options: {" ACT_TYPE_OPTIONS "}.",
             py::arg("type"), py::arg("args"))
        .def("prediction", prediction1,
             "Sets the prediction type. Options: {" PRED_TYPE_OPTIONS "}.",
             py::arg("type"))
        .def("prediction", prediction2,
             "Sets the prediction type and args. Options: {" PRED_TYPE_OPTIONS
             "}.",
             py::arg("type"), py::arg("args"))
        .def("fit", fit1,
             "Creates/updates an action set for a given (state, action, "
             "reward). state shape must be: (x_dim, ).",
             py::arg("state"), py::arg("action"), py::arg("reward"))
        .def("fit", fit2,
             "Executes MAX_TRIALS number of XCSF learning iterations using the "
             "provided training data. X_train shape must be: (n_samples, "
             "x_dim). y_train shape must be: (n_samples, y_dim).",
             py::arg("X_train"), py::arg("y_train"), py::arg("shuffle") = true)
        .def("fit", fit3,
             "Executes MAX_TRIALS number of XCSF learning iterations using the "
             "provided training data and test iterations using the test data. "
             "X_train shape must be: (n_samples, x_dim). y_train shape must "
             "be: (n_samples, y_dim). X_test shape must be: (n_samples, "
             "x_dim). y_test shape must be: (n_samples, y_dim).",
             py::arg("X_train"), py::arg("y_train"), py::arg("X_test"),
             py::arg("y_test"), py::arg("shuffle") = true)
        .def("score", score1,
             "Returns the error using at most N random samples from the "
             "provided data. X_val shape must be: (n_samples, x_dim). y_val "
             "shape must be: (n_samples, y_dim).",
             py::arg("X_val"), py::arg("y_val"), py::arg("N") = 0)
        .def("score", score2,
             "Returns the error using at most N random samples from the "
             "provided data. X_val shape must be: (n_samples, x_dim). y_val "
             "shape must be: (n_samples, y_dim).",
             py::arg("X_val"), py::arg("y_val"), py::arg("N") = 0,
             py::arg("cover"))
        .def("error", error1,
             "Returns a moving average of the system error, updated with step "
             "size BETA.")
        .def("error", error2,
             "Returns the reinforcement learning system prediction error.",
             py::arg("reward"), py::arg("done"), py::arg("max_p"))
        .def("predict", predict1,
             "Returns the XCSF prediction array for the provided input. X_test "
             "shape must be: (n_samples, x_dim). Returns an array of shape: "
             "(n_samples, y_dim). Covering will be invoked for samples where "
             "the match set is empty.",
             py::arg("X_test"))
        .def("predict", predict2,
             "Returns the XCSF prediction array for the provided input. X_test "
             "shape must be: (n_samples, x_dim). Returns an array of shape: "
             "(n_samples, y_dim). If the match set is empty for a sample, the "
             "value of the cover array will be used instead of covering. "
             "cover must be an array of shape: y_dim.",
             py::arg("X_test"), py::arg("cover"))
        .def("save", &XCS::save,
             "Saves the current state of XCSF to persistent storage.",
             py::arg("filename"))
        .def("load", &XCS::load,
             "Loads the current state of XCSF from persistent storage.",
             py::arg("filename"))
        .def("store", &XCS::store,
             "Stores the current XCSF population in memory for later "
             "retrieval, overwriting any previously stored population.")
        .def("retrieve", &XCS::retrieve,
             "Retrieves the previously stored XCSF population from memory.")
        .def("version_major", &XCS::version_major,
             "Returns the version major number.")
        .def("version_minor", &XCS::version_minor,
             "Returns the version minor number.")
        .def("version_build", &XCS::version_build,
             "Returns the version build number.")
        .def("init_trial", &XCS::init_trial, "Initialises a multi-step trial.")
        .def("end_trial", &XCS::end_trial, "Ends a multi-step trial.")
        .def("init_step", &XCS::init_step,
             "Initialises a step in a multi-step trial.")
        .def("end_step", &XCS::end_step, "Ends a step in a multi-step trial.")
        .def("decision", &XCS::decision,
             "Constructs the match set and selects an action to perform for "
             "reinforcement learning. state shape must be: (x_dim, )",
             py::arg("state"), py::arg("explore"))
        .def("update", &XCS::update,
             "Creates the action set using the previously selected action.",
             py::arg("reward"), py::arg("done"))
        .def("seed", &XCS::seed, "Sets the random number seed.",
             py::arg("seed"))
        .def_property("OMP_NUM_THREADS", &XCS::get_omp_num_threads,
                      &XCS::set_omp_num_threads, "Number of CPU cores to use.")
        .def_property("POP_INIT", &XCS::get_pop_init, &XCS::set_pop_init,
                      "Whether to seed the population with random rules.")
        .def_property("MAX_TRIALS", &XCS::get_max_trials, &XCS::set_max_trials,
                      "Number of problem instances to run in one experiment.")
        .def_property("PERF_TRIALS", &XCS::get_perf_trials,
                      &XCS::set_perf_trials,
                      "Number of problem instances to avg performance output.")
        .def_property("POP_SIZE", &XCS::get_pop_max_size,
                      &XCS::set_pop_max_size,
                      "Maximum number of micro-classifiers in the population.")
        .def_property(
            "LOSS_FUNC", &XCS::get_loss_func, &XCS::set_loss_func,
            "Which loss/error function to apply. Options: {" LOSS_OPTIONS "}.")
        .def_property("HUBER_DELTA", &XCS::get_huber_delta,
                      &XCS::set_huber_delta,
                      "Delta parameter for Huber loss calculation.")
        .def_property(
            "ALPHA", &XCS::get_alpha, &XCS::set_alpha,
            "Linear coefficient used to calculate classifier accuracy.")
        .def_property(
            "BETA", &XCS::get_beta, &XCS::set_beta,
            "Learning rate for updating error, fitness, and set size.")
        .def_property("DELTA", &XCS::get_delta, &XCS::set_delta,
                      "Fraction of population to increase deletion vote.")
        .def_property(
            "E0", &XCS::get_e0, &XCS::set_e0,
            "Target error under which classifier accuracy is set to 1.")
        .def_property("STATEFUL", &XCS::get_stateful, &XCS::set_stateful,
                      "Whether classifiers should retain state across trials.")
        .def_property("COMPACTION", &XCS::get_compaction, &XCS::set_compaction,
                      "if sys err < E0: largest of 2 roulette spins deleted.")
        .def_property("INIT_ERROR", &XCS::get_init_error, &XCS::set_init_error,
                      "Initial classifier error value.")
        .def_property("INIT_FITNESS", &XCS::get_init_fitness,
                      &XCS::set_init_fitness,
                      "Initial classifier fitness value.")
        .def_property("NU", &XCS::get_nu, &XCS::set_nu,
                      "Exponent used in calculating classifier accuracy.")
        .def_property("M_PROBATION", &XCS::get_m_probation,
                      &XCS::set_m_probation,
                      "Trials since creation a cl must match at least 1 input.")
        .def_property("THETA_DEL", &XCS::get_theta_del, &XCS::set_theta_del,
                      "Min experience before fitness used during deletion.")
        .def_property(
            "THETA_SUB", &XCS::get_theta_sub, &XCS::set_theta_sub,
            "Minimum experience of a classifier to become a subsumer.")
        .def_property("SET_SUBSUMPTION", &XCS::get_set_subsumption,
                      &XCS::set_set_subsumption,
                      "Whether to perform match set subsumption.")
        .def_property("TELETRANSPORTATION", &XCS::get_teletransportation,
                      &XCS::set_teletransportation,
                      "Maximum steps for a multi-step problem.")
        .def_property("GAMMA", &XCS::get_gamma, &XCS::set_gamma,
                      "Discount factor for multi-step reward.")
        .def_property("P_EXPLORE", &XCS::get_p_explore, &XCS::set_p_explore,
                      "Probability of exploring vs. exploiting.")
        .def_property("EA_SELECT_TYPE", &XCS::get_ea_select_type,
                      &XCS::set_ea_select_type,
                      "EA parental selection type. Options: {" EA_SELECT_OPTIONS
                      "}.")
        .def_property("EA_SELECT_SIZE", &XCS::get_ea_select_size,
                      &XCS::set_ea_select_size,
                      "Fraction of set size for tournaments.")
        .def_property("THETA_EA", &XCS::get_theta_ea, &XCS::set_theta_ea,
                      "Average match set time between EA invocations.")
        .def_property("P_CROSSOVER", &XCS::get_p_crossover,
                      &XCS::set_p_crossover,
                      "Probability of applying crossover.")
        .def_property("LAMBDA", &XCS::get_lambda, &XCS::set_lambda,
                      "Number of offspring to create each EA invocation.")
        .def_property("ERR_REDUC", &XCS::get_err_reduc, &XCS::set_err_reduc,
                      "Amount to reduce an offspring's error.")
        .def_property("FIT_REDUC", &XCS::get_fit_reduc, &XCS::set_fit_reduc,
                      "Amount to reduce an offspring's fitness.")
        .def_property("EA_SUBSUMPTION", &XCS::get_ea_subsumption,
                      &XCS::set_ea_subsumption,
                      "Whether to try and subsume offspring classifiers.")
        .def_property("EA_PRED_RESET", &XCS::get_ea_pred_reset,
                      &XCS::set_ea_pred_reset,
                      "Whether to reset or copy offspring predictions.")
        .def("time", &XCS::get_time, "Returns the current EA time.")
        .def("x_dim", &XCS::get_x_dim, "Returns the x_dim.")
        .def("y_dim", &XCS::get_y_dim, "Returns the y_dim.")
        .def("n_actions", &XCS::get_n_actions, "Returns the number of actions.")
        .def("pset_size", &XCS::get_pset_size,
             "Returns the number of macro-classifiers in the population.")
        .def("pset_num", &XCS::get_pset_num,
             "Returns the number of micro-classifiers in the population.")
        .def("pset_mean_cond_size", &XCS::get_pset_mean_cond_size,
             "Returns the average condition size of classifiers in the "
             "population.")
        .def("pset_mean_pred_size", &XCS::get_pset_mean_pred_size,
             "Returns the average prediction size of classifiers in the "
             "population.")
        .def("pset_mean_pred_eta", &XCS::get_pset_mean_pred_eta,
             "Returns the mean eta for a prediction layer.", py::arg("layer"))
        .def("pset_mean_pred_neurons", &XCS::get_pset_mean_pred_neurons,
             "Returns the mean number of neurons for a prediction layer.",
             py::arg("layer"))
        .def("pset_mean_pred_layers", &XCS::get_pset_mean_pred_layers,
             "Returns the mean number of layers in the prediction networks.")
        .def("pset_mean_pred_connections", &XCS::get_pset_mean_pred_connections,
             "Returns the mean number of connections for a prediction layer.",
             py::arg("layer"))
        .def("pset_mean_cond_neurons", &XCS::get_pset_mean_cond_neurons,
             "Returns the mean number of neurons for a condition layer.",
             py::arg("layer"))
        .def("pset_mean_cond_layers", &XCS::get_pset_mean_cond_layers,
             "Returns the mean number of layers in the condition networks.")
        .def("pset_mean_cond_connections", &XCS::get_pset_mean_cond_connections,
             "Returns the mean number of connections for a condition layer.",
             py::arg("layer"))
        .def("mset_size", &XCS::get_mset_size,
             "Returns the average match set size.")
        .def("aset_size", &XCS::get_aset_size,
             "Returns the average action set size.")
        .def("mfrac", &XCS::get_mfrac,
             "Returns the mean fraction of inputs matched by the best rule.")
        .def("print_pset", &XCS::print_pset, "Prints the current population.",
             py::arg("condition") = true, py::arg("action") = true,
             py::arg("prediction") = true)
        .def("print_params", &XCS::print_params,
             "Prints the XCSF parameters and their current values.")
        .def("pred_expand", &XCS::pred_expand,
             "Inserts a new hidden layer before the output layer within all "
             "prediction neural networks in the population.")
        .def("ae_to_classifier", &XCS::ae_to_classifier,
             "Switches from autoencoding to classification.", py::arg("y_dim"),
             py::arg("n_del"))
        .def("json", &XCS::json_export,
             "Returns a JSON formatted string representing the population set.",
             py::arg("condition") = true, py::arg("action") = true,
             py::arg("prediction") = true)
        .def("json_write", &XCS::json_write,
             "Writes the current population set to a file in JSON.",
             py::arg("filename"))
        .def("json_read", &XCS::json_read,
             "Reads classifiers from a JSON file and adds to the population.",
             py::arg("filename"))
        .def("json_parameters", &XCS::json_parameters,
             "Returns a JSON formatted string representing the parameters.")
        .def("json_insert_cl", &XCS::json_insert_cl,
             "Creates a classifier from JSON and inserts into the population.",
             py::arg("json_str"))
        .def("json_insert", &XCS::json_insert,
             "Creates classifiers from JSON and inserts into the population.",
             py::arg("json_str"));
}
