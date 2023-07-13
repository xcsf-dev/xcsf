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

#include <cstdio>
#include <fstream>
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
    bool first_fit; //!< Whether this is the first execution of fit()
    py::dict params; //!< Dictionary of parameters and their values

  public:
    /**
     * @brief Default Constructor.
     */
    XCS()
    {
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
        first_fit = true;
        param_init(&xcs, 1, 1, 1);
        update_params();
    }

    /**
     * @brief Constructor.
     * @param [in] kwargs Parameters and their values.
     */
    XCS(py::kwargs kwargs) : XCS()
    {
        set_params(kwargs);
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
        if (first_fit) {
            first_fit = false;
            xcsf_init(&xcs);
        }
        return xcs_rl_fit(&xcs, state, action, reward);
    }

    /**
     * @brief Initialises a reinforcement learning trial.
     */
    void
    init_trial(void)
    {
        if (first_fit) {
            first_fit = false;
            xcsf_init(&xcs);
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
     * @param [in] first_fit Whether this is the first call to fit().
     */
    void
    load_input(struct Input *data, const py::array_t<double> X,
               const py::array_t<double> Y, const bool first_fit)
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
        if (first_fit) { // automatically set x_dim, y_dim, n_actions
            set_dims(buf_x.ndim, buf_x.shape[1], buf_y.ndim, buf_y.shape[1]);
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
     * @brief Executes MAX_TRIALS number of XCSF learning iterations using the
     * provided training data.
     * @param [in] train_X The input values to use for training.
     * @param [in] train_Y The true output values to use for training.
     * @param [in] shuffle Whether to randomise the instances during training.
     * @param [in] warm_start Whether to continue with existing population.
     * @return The fitted XCSF model.
     */
    XCS &
    fit(const py::array_t<double> train_X, const py::array_t<double> train_Y,
        const bool shuffle, const bool warm_start)
    {
        load_input(train_data, train_X, train_Y, first_fit);
        if (first_fit) {
            first_fit = false;
            xcsf_init(&xcs);
        } else if (!warm_start) {
            xcsf_free(&xcs);
            xcsf_init(&xcs);
        }
        xcs_supervised_fit(&xcs, train_data, NULL, shuffle);
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
     * @brief Returns the XCSF prediction array for the provided input.
     * If the match set is empty, the prediction array will be zeros.
     * @param [in] X The input variables.
     * @return The prediction array values.
     */
    py::array_t<double>
    predict(const py::array_t<double> X)
    {
        double *cov = (double *) calloc(xcs.x_dim, sizeof(double));
        py::array_t<double> predictions = get_predictions(X, cov);
        free(cov);
        return predictions;
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
        load_input(test_data, X, Y, false);
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
        double *cov = (double *) calloc(xcs.x_dim, sizeof(double));
        double score = get_score(X, Y, N, cov);
        free(cov);
        return score;
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
     * @return Parameter dictionary.
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

    double (XCS::*fit1)(const py::array_t<double>, const int, const double) =
        &XCS::fit;
    XCS &(XCS::*fit2)(const py::array_t<double>, const py::array_t<double>,
                      const bool, const bool) = &XCS::fit;

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
             py::arg("warm_start") = false)
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
        .def("time", &XCS::get_time, "Returns the current EA time.")
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
        .def(py::pickle(
            [](const XCS &obj) { return obj.serialize(); },
            [](const py::bytes &state) { return XCS::deserialize(state); }));
}
