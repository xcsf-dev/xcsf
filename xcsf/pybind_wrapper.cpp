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

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

#include "pybind_callback.h"
#include "pybind_callback_checkpoint.h"
#include "pybind_callback_earlystop.h"
#include "pybind_utils.h"

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
    struct Input *val_data; //!< Validation data
    py::dict params; //!< Dictionary of parameters and their values
    py::list metric_train;
    py::list metric_val;
    py::list metric_trial;
    py::list metric_psize;
    py::list metric_msize;
    py::list metric_mfrac;
    int metric_counter;

  public:
    /**
     * @brief Default Constructor.
     */
    XCS()
    {
        reset();
        xcsf_init(&xcs);
    }

    /**
     * @brief Constructor.
     * @param [in] kwargs Parameters and their values.
     */
    explicit XCS(py::kwargs kwargs)
    {
        reset();
        set_params(kwargs);
        xcsf_init(&xcs);
    }

    /**
     * @brief Resets basic constructor variables.
     */
    void
    reset(void)
    {
        state = NULL;
        action = 0;
        payoff = 0;
        train_data = new struct Input;
        train_data->n_samples = 0;
        train_data->x_dim = 0;
        train_data->y_dim = 0;
        train_data->x = NULL;
        train_data->y = NULL;
        test_data = new struct Input;
        test_data->n_samples = 0;
        test_data->x_dim = 0;
        test_data->y_dim = 0;
        test_data->x = NULL;
        test_data->y = NULL;
        val_data = NULL;
        metric_counter = 0;
        param_init(&xcs, 1, 1, 1);
        update_params();
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
        size_t s = xcsf_load(&xcs, filename);
        update_params();
        return s;
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
     * @brief Sets XCSF input and output dimensions.
     * @param [in] n_x_dim Number of input dimensions.
     * @param [in] x1 Size of second input dimension.
     * @param [in] n_y_dim Number of output dimensions.
     * @param [in] y1 Size of second output dimension.
     */
    void
    set_dims(int n_x_dim, int x1, int n_y_dim, int y1)
    {
        py::dict kwargs;
        kwargs["n_actions"] = 1;
        if (n_x_dim > 1) {
            kwargs["x_dim"] = x1;
        } else {
            kwargs["x_dim"] = 1;
        }
        if (n_y_dim > 1) {
            kwargs["y_dim"] = y1;
        } else {
            kwargs["y_dim"] = 1;
        }
        // update external params dict
        for (const auto &item : kwargs) {
            params[item.first] = item.second;
        }
        // flush param update to make sure neural nets resize
        set_params(params);
    }

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
        if (buf_x.ndim < 1 || buf_x.ndim > 2) {
            std::string error = "load_input(): X must be 1 or 2-D array";
            throw std::invalid_argument(error);
        }
        if (buf_y.ndim < 1 || buf_y.ndim > 2) {
            std::string error = "load_input(): Y must be 1 or 2-D array";
            throw std::invalid_argument(error);
        }
        if (buf_x.shape[0] != buf_y.shape[0]) {
            std::string error = "load_input(): X and Y n_samples are not equal";
            throw std::invalid_argument(error);
        }
        if (buf_x.ndim > 1 && buf_x.shape[1] != xcs.x_dim) {
            std::ostringstream error;
            error << "load_input():";
            error << " received x_dim: (" << buf_x.shape[1] << ")";
            error << " but expected (" << xcs.x_dim << ")" << std::endl;
            error << "Perhaps reshape your data.";
            throw std::invalid_argument(error.str());
        }
        if (buf_y.ndim > 1 && buf_y.shape[1] != xcs.y_dim) {
            std::ostringstream error;
            error << "load_input():";
            error << " received y_dim: (" << buf_y.shape[1] << ")";
            error << " but expected (" << xcs.y_dim << ")" << std::endl;
            error << "Perhaps reshape your data.";
            throw std::invalid_argument(error.str());
        }
        data->n_samples = buf_x.shape[0];
        data->x_dim = xcs.x_dim;
        data->y_dim = xcs.y_dim;
        data->x = (double *) buf_x.ptr;
        data->y = (double *) buf_y.ptr;
    }

    /**
     * @brief Prints the current performance metrics.
     */
    void
    print_status()
    {
        double trial = py::cast<double>(metric_trial[metric_trial.size() - 1]);
        double train = py::cast<double>(metric_train[metric_train.size() - 1]);
        double psize = py::cast<double>(metric_psize[metric_psize.size() - 1]);
        double msize = py::cast<double>(metric_msize[metric_msize.size() - 1]);
        double mfrac = py::cast<double>(metric_mfrac[metric_mfrac.size() - 1]);
        std::ostringstream status;
        status << get_timestamp();
        status << " trials=" << trial;
        status << " train=" << std::fixed << std::setprecision(5) << train;
        if (val_data != NULL) {
            double val = py::cast<double>(metric_val[metric_val.size() - 1]);
            status << " val=" << std::fixed << std::setprecision(5) << val;
        }
        status << " pset=" << std::fixed << std::setprecision(1) << psize;
        status << " mset=" << std::fixed << std::setprecision(1) << msize;
        status << " mfrac=" << std::fixed << std::setprecision(2) << mfrac;
        py::print(status.str());
    }

    /**
     * @brief Updates performance metrics.
     * @param [in] train The current training error.
     * @param [in] val The current validation error.
     * @param [in] n_trials Number of trials run.
     */
    void
    update_metrics(const double train, const double val, const int n_trials)
    {
        const int trial = (1 + metric_counter) * n_trials;
        metric_train.append(train);
        metric_val.append(val);
        metric_trial.append(trial);
        metric_psize.append(xcs.pset.size);
        metric_msize.append(xcs.mset_size);
        metric_mfrac.append(xcs.mfrac);
        ++metric_counter;
    }

    /**
     * @brief Loads validation data if present in kwargs.
     * @param [in] kwargs Parameters and their values.
     */
    void
    load_validation_data(py::kwargs kwargs)
    {
        val_data = NULL;
        if (kwargs.contains("validation_data")) {
            py::tuple data = kwargs["validation_data"].cast<py::tuple>();
            if (data) {
                py::array_t<double> X_val = data[0].cast<py::array_t<double>>();
                py::array_t<double> y_val = data[1].cast<py::array_t<double>>();
                load_input(test_data, X_val, y_val);
                val_data = test_data;
                // use zeros for validation predictions instead of covering
                memset(xcs.cover, 0, sizeof(double) * xcs.pa_size);
            }
        }
    }

    /**
     * @brief Executes callbacks and returns whether to terminate.
     * @param [in] callbacks The callbacks to perform.
     * @return Whether to terminate early.
     */
    bool
    callbacks_run(py::list callbacks)
    {
        bool terminate = false;
        py::dict metrics = get_metrics();
        for (py::handle item : callbacks) {
            if (py::isinstance<Callback>(item)) {
                Callback *cb = py::cast<Callback *>(item);
                if (cb->run(&xcs, metrics)) {
                    terminate = true;
                }
            } else {
                std::ostringstream err;
                err << "unsupported callback" << std::endl;
                throw std::invalid_argument(err.str());
            }
        }
        return terminate;
    }

    /**
     * @brief Executes callback finish.
     * @param [in] callbacks The callbacks to perform.
     */
    void
    callbacks_finish(py::list callbacks)
    {
        for (py::handle item : callbacks) {
            if (py::isinstance<Callback>(item)) {
                Callback *cb = py::cast<Callback *>(item);
                cb->finish(&xcs);
            } else {
                std::ostringstream err;
                err << "unsupported callback" << std::endl;
                throw std::invalid_argument(err.str());
            }
        }
    }

    /**
     * @brief Executes MAX_TRIALS number of XCSF learning iterations using the
     * provided training data.
     * @param [in] X_train The input values to use for training.
     * @param [in] y_train The true output values to use for training.
     * @param [in] shuffle Whether to randomise the instances during training.
     * @param [in] warm_start Whether to continue with existing population.
     * @param [in] verbose Whether to print learning metrics.
     * @param [in] callbacks List of Callback objects or None.
     * @param [in] kwargs Keyword arguments.
     * @return The fitted XCSF model.
     */
    XCS &
    fit(const py::array_t<double> X_train, const py::array_t<double> y_train,
        const bool shuffle, const bool warm_start, const bool verbose,
        py::object callbacks, py::kwargs kwargs)
    {
        if (!warm_start) { // re-initialise XCSF as necessary
            xcsf_free(&xcs);
            xcsf_init(&xcs);
        }
        load_input(train_data, X_train, y_train);
        load_validation_data(kwargs);
        // get callbacks
        py::list calls;
        if (py::isinstance<py::list>(callbacks)) {
            calls = callbacks.cast<py::list>();
        }
        // break up the learning into epochs to track metrics
        const int n = ceil(xcs.MAX_TRIALS / (double) xcs.PERF_TRIALS);
        const int n_trials = std::min(xcs.MAX_TRIALS, xcs.PERF_TRIALS);
        for (int i = 0; i < n; ++i) {
            const double train =
                xcs_supervised_fit(&xcs, train_data, NULL, shuffle, n_trials);
            double val = 0;
            if (val_data != NULL) {
                val = xcs_supervised_score(&xcs, val_data, xcs.cover);
            }
            update_metrics(train, val, n_trials);
            if (verbose) {
                print_status();
            }
            if (callbacks_run(calls)) {
                break;
            }
        }
        callbacks_finish(calls);
        return *this;
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
     * @brief Sets the XCSF cover array to values given, or zeros.
     * @param [in] cover The values to use instead of covering.
     */
    void
    set_cover(const py::object &cover)
    {
        if (cover.is_none()) {
            memset(xcs.cover, 0, sizeof(double) * xcs.pa_size);
        } else {
            py::array_t<double> cover_arr = cover.cast<py::array_t<double>>();
            xcs.cover = get_cover(cover_arr);
        }
    }

    /**
     * @brief Returns the XCSF prediction array for the provided input.
     * @param [in] X The input variables.
     * @param [in] cover If the match set is empty, the prediction array will
     * be set to this value instead of covering.
     * @return The prediction array values.
     */
    py::array_t<double>
    predict(const py::array_t<double> X, const py::object &cover)
    {
        const py::buffer_info buf_x = X.request();
        if (buf_x.ndim < 1 || buf_x.ndim > 2) {
            std::string error = "predict(): X must be 1 or 2-D array";
            throw std::invalid_argument(error);
        }
        if (buf_x.ndim > 1 && buf_x.shape[1] != xcs.x_dim) {
            std::ostringstream error;
            error << "predict():";
            error << " received x_dim: (" << buf_x.shape[1] << ")";
            error << " but expected (" << xcs.x_dim << ")" << std::endl;
            error << "Perhaps reshape your data.";
            throw std::invalid_argument(error.str());
        }
        const int n_samples = buf_x.shape[0];
        const double *input = reinterpret_cast<double *>(buf_x.ptr);
        double *output =
            (double *) malloc(sizeof(double) * n_samples * xcs.pa_size);
        set_cover(cover);
        xcs_supervised_predict(&xcs, input, output, n_samples, xcs.cover);
        return py::array_t<double>(
            std::vector<ptrdiff_t>{ n_samples, xcs.pa_size }, output);
    }

    /**
     * @brief Returns the error using N random samples from the provided data.
     * @param [in] X The input values to use for scoring.
     * @param [in] Y The true output values to use for scoring.
     * @param [in] N The maximum number of samples to draw randomly for scoring.
     * @param [in] cover If the match set is empty, the prediction array will
     * be set to this value, otherwise it is set to zeros.
     * @return The average XCSF error using the loss function.
     */
    double
    score(const py::array_t<double> X, const py::array_t<double> Y, const int N,
          const py::object &cover)
    {
        set_cover(cover);
        load_input(test_data, X, Y);
        if (N > 1) {
            return xcs_supervised_score_n(&xcs, test_data, N, xcs.cover);
        }
        return xcs_supervised_score(&xcs, test_data, xcs.cover);
    }

    /**
     * @brief Implements pickle file writing.
     * @details Uses a temporary binary file.
     * @return The pickled XCSF.
     */
    py::bytes
    serialize() const
    {
        // Write XCSF to a temporary binary file
        const char *filename = "_tmp_pickle.bin";
        xcsf_save(&xcs, filename);
        // Read the binary file into bytes
        std::ifstream file(filename, std::ios::binary);
        std::string state((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
        file.close();
        // Delete the temporary file
        if (std::remove(filename) != 0) {
            perror("Error deleting temporary pickle file");
        }
        // Return the binary data as bytes
        return py::bytes(state);
    }

    /**
     * @brief Implements pickle file reading.
     * @details Uses a temporary binary file.
     * @param state The pickled state of a saved XCSF.
     */
    static XCS
    deserialize(const py::bytes &state)
    {
        // Write the XCSF bytes to a temporary binary file
        const char *filename = "_tmp_pickle.bin";
        std::ofstream file(filename, std::ios::binary);
        file.write(state.cast<std::string>().c_str(),
                   state.cast<std::string>().size());
        file.close();
        // Create a new XCSF instance
        XCS xcs = XCS();
        // Load XCSF
        xcsf_load(&xcs.xcs, filename);
        // Update object params
        xcs.update_params();
        // Delete the temporary file
        if (std::remove(filename) != 0) {
            perror("Error deleting temporary pickle file");
        }
        // Return the deserialized XCSF
        return xcs;
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

    py::dict
    get_metrics(void)
    {
        py::dict metrics;
        metrics["train"] = metric_train;
        metrics["val"] = metric_val;
        metrics["trials"] = metric_trial;
        metrics["psize"] = metric_psize;
        metrics["msize"] = metric_msize;
        metrics["mfrac"] = metric_mfrac;
        return metrics;
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
     * @brief Updates the Python object's parameter dictionary.
     */
    void
    update_params()
    {
        char *json_str = param_json_export(&xcs);
        py::module json = py::module::import("json");
        py::object parsed_json = json.attr("loads")(json_str);
        py::dict result(parsed_json);
        params = result;
        free(json_str);
    }

    /**
     * @brief Returns a dictionary of parameters.
     * @param deep For sklearn compatibility.
     * @return External parameter dictionary.
     */
    py::dict
    get_params(const bool deep)
    {
        (void) deep;
        return params;
    }

    /**
     * @brief Sets parameter values.
     * @param kwargs Parameters and their values.
     * @return The XCSF object.
     */
    XCS &
    set_params(py::kwargs kwargs)
    {
        py::dict kwargs_dict(kwargs);
        py::module json_module = py::module::import("json");
        py::object json_dumps = json_module.attr("dumps")(kwargs_dict);
        std::string json_str = json_dumps.cast<std::string>();
        const char *json_params = json_str.c_str();
        param_json_import(&xcs, json_params);
        for (const auto &item : kwargs_dict) {
            params[item.first] = item.second;
        }
        return *this;
    }

    /**
     * @brief Returns a dictionary of the internal parameters.
     * @return Internal parameter dictionary.
     */
    py::dict
    internal_params()
    {
        char *json_str = param_json_export(&xcs);
        py::module json_module = py::module::import("json");
        py::dict internal_params = json_module.attr("loads")(json_str);
        free(json_str);
        return internal_params;
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
        clset_json_insert_cl(&xcs, json);
        cJSON_Delete(json);
    }

    /**
     * @brief Creates classifiers from JSON and inserts into the population.
     * @param [in] json_str JSON formatted string representing a classifier.
     */
    void
    json_insert(const std::string &json_str)
    {
        clset_json_insert(&xcs, json_str.c_str());
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

    double (XCS::*fit1)(const py::array_t<double>, const int, const double) =
        &XCS::fit;
    XCS &(XCS::*fit2)(const py::array_t<double>, const py::array_t<double>,
                      const bool, const bool, const bool, py::object,
                      py::kwargs) = &XCS::fit;

    double (XCS::*error1)(void) = &XCS::error;
    double (XCS::*error2)(const double, const bool, const double) = &XCS::error;

    py::class_<Callback, std::unique_ptr<Callback, py::nodelete>>(m,
                                                                  "Callback");

    py::class_<EarlyStoppingCallback, Callback,
               std::unique_ptr<EarlyStoppingCallback, py::nodelete>>(
        m, "EarlyStoppingCallback")
        .def(py::init<py::str, int, bool, double, int, bool>(),
             "Creates a callback for terminating the fit function early.",
             py::arg("monitor") = "train", py::arg("patience") = 0,
             py::arg("restore_best") = false, py::arg("min_delta") = 0,
             py::arg("start_from") = 0, py::arg("verbose") = true);

    py::class_<CheckpointCallback, Callback,
               std::unique_ptr<CheckpointCallback, py::nodelete>>(
        m, "CheckpointCallback")
        .def(py::init<py::str, std::string, bool, int, bool>(),
             "Creates a callback for automatically saving XCSF.",
             py::arg("monitor") = "train", py::arg("filename") = "xcsf.bin",
             py::arg("save_best_only") = false, py::arg("save_freq") = 0,
             py::arg("verbose") = true);

    py::class_<XCS>(m, "XCS")
        .def(py::init(), "Creates a new XCSF class with default arguments.")
        .def(py::init<py::kwargs>(),
             "Creates a new XCSF class with specified arguments.")
        .def("fit", fit1,
             "Creates/updates an action set for a given (state, action, "
             "reward). state shape must be: (x_dim, ).",
             py::arg("state"), py::arg("action"), py::arg("reward"))
        .def("fit", fit2,
             "Executes MAX_TRIALS number of XCSF learning iterations using the "
             "provided training data. X_train shape must be: (n_samples, "
             "x_dim). y_train shape must be: (n_samples, y_dim).",
             py::arg("X_train"), py::arg("y_train"), py::arg("shuffle") = true,
             py::arg("warm_start") = false, py::arg("verbose") = true,
             py::arg("callbacks") = py::none())
        .def(
            "score", &XCS::score,
            "Returns the error using at most N random samples from the "
            "provided data. N=0 uses all. X shape must be: (n_samples, x_dim). "
            "y shape must be: (n_samples, y_dim). If the match set is empty "
            "for a sample, the value of the cover array will be used "
            "otherwise zeros.",
            py::arg("X"), py::arg("y"), py::arg("N") = 0,
            py::arg("cover") = py::none())
        .def("error", error1,
             "Returns a moving average of the system error, updated with step "
             "size BETA.")
        .def("error", error2,
             "Returns the reinforcement learning system prediction error.",
             py::arg("reward"), py::arg("done"), py::arg("max_p"))
        .def("predict", &XCS::predict,
             "Returns the XCSF prediction array for the provided input. X "
             "shape must be: (n_samples, x_dim). Returns an array of shape: "
             "(n_samples, y_dim). If the match set is empty for a sample, the "
             "value of the cover array will be used, otherwise zeros. "
             "Cover must be an array of shape: y_dim.",
             py::arg("X"), py::arg("cover") = py::none())
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
        .def("time", &XCS::get_time, "Returns the current EA time.")
        .def("get_metrics", &XCS::get_metrics,
             "Returns a dictionary of performance metrics.")
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
        .def("get_params", &XCS::get_params, py::arg("deep") = true,
             "Returns a dictionary of parameters and their values.")
        .def("set_params", &XCS::set_params, "Sets parameters.")
        .def("json_insert_cl", &XCS::json_insert_cl,
             "Creates a classifier from JSON and inserts into the population.",
             py::arg("json_str"))
        .def("json_insert", &XCS::json_insert,
             "Creates classifiers from JSON and inserts into the population.",
             py::arg("json_str"))
        .def("internal_params", &XCS::internal_params, "Gets internal params.")
        .def(py::pickle(
            [](const XCS &obj) { return obj.serialize(); },
            [](const py::bytes &state) { return XCS::deserialize(state); }));
}
