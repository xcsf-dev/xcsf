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

#include "../pybind11/include/pybind11/pybind11.h"
#include "../pybind11/include/pybind11/numpy.h"
#include <string>
#include <vector>

namespace py = pybind11;

extern "C" {   
#include <stdbool.h>
#include "xcsf.h"
#include "xcs_single_step.h"
#include "pa.h"
#include "config.h"
#include "utils.h"
#include "loss.h"
#include "clset.h"

#ifdef PARALLEL
#include <omp.h>
#endif
}

void xcs_init(const char *filename);

class XCS
{
    private:
        XCSF xcs; //!< XCSF data structure
        double *state; //!< Current input state for RL
        int action; //!< Current action for RL
        INPUT *train_data; //!< Current training data for supervised learning
        INPUT *test_data; //!< Currrent test data for supervised learning

    public:
        /**
         * @brief Constructor for single-step reinforcement learning.
         */
        XCS(int x_dim, int n_actions, _Bool multistep)
        {
            (void)multistep; // not yet implemented for python
            xcs.x_dim = x_dim;
            xcs.y_dim = 1;
            xcs.n_actions = n_actions;
            xcs_init("default.ini");
            pa_init(&xcs);
            xcs_single_init(&xcs);
        }

        /**
         * @brief Constructor for supervised learning with default config.
         */
        XCS(int x_dim, int y_dim) : XCS(x_dim, y_dim, "default.ini") {}

        /**
         * @brief Constructor for supervised learning with a specified config.
         */
        XCS(int x_dim, int y_dim, const char *filename)
        {
            xcs.x_dim = x_dim;
            xcs.y_dim = y_dim;
            xcs.n_actions = 1;
            xcs_init(filename);
        }

        /**
         * @brief Initialises python XCS structure.
         */
        void xcs_init(const char *filename)
        {
            config_init(&xcs, filename);
            xcsf_init(&xcs);
            state = NULL;
            action = 0;
            train_data = (INPUT*)malloc(sizeof(INPUT));
            train_data->n_samples = 0;
            train_data->x_dim = 0;
            train_data->y_dim = 0;
            train_data->x = NULL;
            train_data->y = NULL;
            test_data = (INPUT*)malloc(sizeof(INPUT));
            test_data->n_samples = 0;
            test_data->x_dim = 0;
            test_data->y_dim = 0;
            test_data->x = NULL;
            test_data->y = NULL;
        }

        double version(){ return xcsf_version(); }
        size_t save(char *fname) { return xcsf_save(&xcs, fname); }
        size_t load(char *fname) { return xcsf_load(&xcs, fname); }
        void print_params() { config_print(&xcs); }

        void print_pop(_Bool printc, _Bool printa, _Bool printp)
        {
            xcsf_print_pop(&xcs, printc, printa, printp);
        }

        /* Reinforcement learning */

        void single_reset()
        {
            if(xcs.time == 0) {
                clset_pop_init(&xcs);
            }
            xcs_single_free(&xcs);
        }

        int single_decision(py::array_t<double> input, _Bool explore)
        {
            py::buffer_info buf = input.request();
            state = (double *) buf.ptr;
            xcs.train = explore;
            action = xcs_single_decision(&xcs, state);
            return action;
        }

        void single_update(double reward)
        {
            xcs_single_update(&xcs, state, action, reward);
        }

        double single_error(double reward)
        {
            return xcs_single_error(&xcs, reward);
        }

        /* Supervised learning */

        double fit(py::array_t<double> train_X, py::array_t<double> train_Y, _Bool shuffle)
        {
            py::buffer_info buf_x = train_X.request();
            py::buffer_info buf_y = train_Y.request();
            if(buf_x.shape[0] != buf_y.shape[0]) {
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
            if(xcs.time == 0) {
                clset_pop_init(&xcs);
            }
            // execute
            return xcsf_fit(&xcs, train_data, NULL, shuffle);
        }

        double fit(py::array_t<double> train_X, py::array_t<double> train_Y,
                py::array_t<double> test_X, py::array_t<double> test_Y, _Bool shuffle)
        {
            py::buffer_info buf_train_x = train_X.request();
            py::buffer_info buf_train_y = train_Y.request();
            py::buffer_info buf_test_x = test_X.request();
            py::buffer_info buf_test_y = test_Y.request();
            if(buf_train_x.shape[0] != buf_train_y.shape[0]) {
                printf("error: training X and Y n_samples are not equal\n");
                exit(EXIT_FAILURE);
            }
            if(buf_test_x.shape[0] != buf_test_y.shape[0]) {
                printf("error: testing X and Y n_samples are not equal\n");
                exit(EXIT_FAILURE);
            }
            if(buf_train_x.shape[1] != buf_test_x.shape[1]) {
                printf("error: number of training and testing X cols are not equal\n");
                exit(EXIT_FAILURE);
            }
            if(buf_train_y.shape[1] != buf_test_y.shape[1]) {
                printf("error: number of training and testing Y cols are not equal\n");
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
            if(xcs.time == 0) {
                clset_pop_init(&xcs);
            }
            // execute
            return xcsf_fit(&xcs, train_data, test_data, shuffle);
        }

        py::array_t<double> predict(py::array_t<double> x)
        {
            // inputs to predict
            py::buffer_info buf_x = x.request();
            int n_samples = buf_x.shape[0];
            double *input = (double *) buf_x.ptr;
            // predicted outputs
            double *output = (double *) malloc(sizeof(double) * n_samples * xcs.y_dim);
            xcsf_predict(&xcs, input, output, n_samples);
            // return numpy array
            return py::array_t<double>(std::vector<ptrdiff_t>{n_samples, xcs.y_dim}, output);
        }

        double score(py::array_t<double> test_X, py::array_t<double> test_Y)
        {
            py::buffer_info buf_x = test_X.request();
            py::buffer_info buf_y = test_Y.request();
            if(buf_x.shape[0] != buf_y.shape[0]) {
                printf("error: training X and Y n_samples are not equal\n");
                exit(EXIT_FAILURE);
            }
            test_data->n_samples = buf_x.shape[0];
            test_data->x_dim = buf_x.shape[1];
            test_data->y_dim = buf_y.shape[1];
            test_data->x = (double *) buf_x.ptr;
            test_data->y = (double *) buf_y.ptr;
            return xcsf_score(&xcs, test_data);
        }

        /* GETTERS */

        py::list get_cond_num_neurons()
        {
            py::list list;
            for(int i = 0; i < MAX_LAYERS && xcs.COND_NUM_NEURONS[i] > 0; i++) {
                list.append(xcs.COND_NUM_NEURONS[i]);
            }
            return list;
        }

        py::list get_cond_max_neurons()
        {
            py::list list;
            for(int i = 0; i < MAX_LAYERS && xcs.COND_MAX_NEURONS[i] > 0; i++) {
                list.append(xcs.COND_MAX_NEURONS[i]);
            }
            return list;
        }

        py::list get_pred_num_neurons()
        {
            py::list list;
            for(int i = 0; i < MAX_LAYERS && xcs.PRED_NUM_NEURONS[i] > 0; i++) {
                list.append(xcs.PRED_NUM_NEURONS[i]);
            }
            return list;
        }

        py::list get_pred_max_neurons()
        {
            py::list list;
            for(int i = 0; i < MAX_LAYERS && xcs.PRED_MAX_NEURONS[i] > 0; i++) {
                list.append(xcs.PRED_MAX_NEURONS[i]);
            }
            return list;
        }

        int get_omp_num_threads() { return xcs.OMP_NUM_THREADS; }
        _Bool get_pop_init() { return xcs.POP_INIT; }
        int get_max_trials() { return xcs.MAX_TRIALS; }
        int get_perf_trials() { return xcs.PERF_TRIALS; }
        int get_pop_max_size() { return xcs.POP_SIZE; }
        int get_loss_func() { return xcs.LOSS_FUNC; }
        double get_alpha() { return xcs.ALPHA; }
        double get_beta() { return xcs.BETA; }
        double get_delta() { return xcs.DELTA; }
        double get_eps_0() { return xcs.EPS_0; }
        double get_err_reduc() { return xcs.ERR_REDUC; }
        double get_fit_reduc() { return xcs.FIT_REDUC; }
        double get_init_error() { return xcs.INIT_ERROR; }
        double get_init_fitness() { return xcs.INIT_FITNESS; }
        double get_nu() { return xcs.NU; }
        int get_m_probation() { return xcs.M_PROBATION; }
        int get_theta_del() { return xcs.THETA_DEL; }
        int get_act_type() { return xcs.ACT_TYPE; }
        int get_cond_type() { return xcs.COND_TYPE; }
        int get_pred_type() { return xcs.PRED_TYPE; }
        double get_p_crossover() { return xcs.P_CROSSOVER; }
        double get_theta_ea() { return xcs.THETA_EA; }
        int get_lambda() { return xcs.LAMBDA; }
        int get_ea_select_type() { return xcs.EA_SELECT_TYPE; }
        double get_ea_select_size() { return xcs.EA_SELECT_SIZE; }
        int get_sam_type() { return xcs.SAM_TYPE; }
        double get_max_con() { return xcs.COND_MAX; }
        double get_min_con() { return xcs.COND_MIN; }
        double get_cond_smin() { return xcs.COND_SMIN; }
        double get_cond_bits() { return xcs.COND_BITS; }
        _Bool get_cond_evolve_weights() { return xcs.COND_EVOLVE_WEIGHTS; }
        _Bool get_cond_evolve_neurons() { return xcs.COND_EVOLVE_NEURONS; }
        _Bool get_cond_evolve_functions() { return xcs.COND_EVOLVE_FUNCTIONS; }
        int get_cond_output_activation() { return xcs.COND_OUTPUT_ACTIVATION; }
        int get_cond_hidden_activation() { return xcs.COND_HIDDEN_ACTIVATION; }
        int get_pred_output_activation() { return xcs.PRED_OUTPUT_ACTIVATION; }
        int get_pred_hidden_activation() { return xcs.PRED_HIDDEN_ACTIVATION; }
        double get_pred_momentum() { return xcs.PRED_MOMENTUM; }
        _Bool get_pred_evolve_weights() { return xcs.PRED_EVOLVE_WEIGHTS; }
        _Bool get_pred_evolve_neurons() { return xcs.PRED_EVOLVE_NEURONS; }
        _Bool get_pred_evolve_functions() { return xcs.PRED_EVOLVE_FUNCTIONS; }
        _Bool get_pred_evolve_eta() { return xcs.PRED_EVOLVE_ETA; }
        _Bool get_pred_sgd_weights() { return xcs.PRED_SGD_WEIGHTS; }
        _Bool get_pred_reset() { return xcs.PRED_RESET; }
        int get_max_neuron_mod() { return xcs.MAX_NEURON_MOD; }
        int get_dgp_num_nodes() { return xcs.DGP_NUM_NODES; }
        _Bool get_reset_states() { return xcs.RESET_STATES; }
        int get_max_k() { return xcs.MAX_K; }
        int get_max_t() { return xcs.MAX_T; }
        int get_gp_num_cons() { return xcs.GP_NUM_CONS; }
        int get_gp_init_depth() { return xcs.GP_INIT_DEPTH; }
        double get_pred_eta() { return xcs.PRED_ETA; }
        double get_cond_eta() { return xcs.COND_ETA; }
        double get_pred_x0() { return xcs.PRED_X0; }
        double get_pred_rls_scale_factor() { return xcs.PRED_RLS_SCALE_FACTOR; }
        double get_pred_rls_lambda() { return xcs.PRED_RLS_LAMBDA; }
        int get_theta_sub() { return xcs.THETA_SUB; }
        _Bool get_ea_subsumption() { return xcs.EA_SUBSUMPTION; }
        _Bool get_set_subsumption() { return xcs.SET_SUBSUMPTION; }
        int get_pop_size() { return xcs.pset.size; }
        int get_pop_num() { return xcs.pset.num; }
        int get_time() { return xcs.time; }
        double get_x_dim() { return xcs.x_dim; }
        double get_y_dim() { return xcs.y_dim; }
        double get_n_actions() { return xcs.n_actions; }
        double get_pop_mean_cond_size() { return clset_mean_cond_size(&xcs, &xcs.pset); }
        double get_pop_mean_pred_size() { return clset_mean_pred_size(&xcs, &xcs.pset); }
        double get_pop_mean_pred_eta(int layer) { return clset_mean_eta(&xcs, &xcs.pset, layer); }
        double get_pop_mean_pred_neurons(int layer){ return clset_mean_neurons(&xcs, &xcs.pset, layer); }
        double get_pop_mean_pred_layers() { return clset_mean_layers(&xcs, &xcs.pset); }
        double get_msetsize() { return xcs.msetsize; }
        double get_mfrac() { return xcs.mfrac; }
        int get_teletransportation() { return xcs.TELETRANSPORTATION; }
        double get_gamma() { return xcs.GAMMA; }
        double get_p_explore() { return xcs.P_EXPLORE; }

        /* SETTERS */

        void set_omp_num_threads(int a)
        {
            xcs.OMP_NUM_THREADS = a; 
#ifdef PARALLEL
            omp_set_num_threads(xcs.OMP_NUM_THREADS);
#endif
        }

        void set_cond_num_neurons(py::list &a)
        {
            memset(xcs.COND_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
            for(size_t i = 0; i < a.size(); i++) {
                xcs.COND_NUM_NEURONS[i] = a[i].cast<int>();
            }
        }

        void set_cond_max_neurons(py::list &a)
        {
            memset(xcs.COND_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
            for(size_t i = 0; i < a.size(); i++) {
                xcs.COND_MAX_NEURONS[i] = a[i].cast<int>();
            }
        }

        void set_pred_num_neurons(py::list &a)
        {
            memset(xcs.PRED_NUM_NEURONS, 0, MAX_LAYERS * sizeof(int));
            for(size_t i = 0; i < a.size(); i++) {
                xcs.PRED_NUM_NEURONS[i] = a[i].cast<int>();
            }
        }

        void set_pred_max_neurons(py::list &a)
        {
            memset(xcs.PRED_MAX_NEURONS, 0, MAX_LAYERS * sizeof(int));
            for(size_t i = 0; i < a.size(); i++) {
                xcs.PRED_MAX_NEURONS[i] = a[i].cast<int>(); 
            }
        }

        void set_pop_init(_Bool a) { xcs.POP_INIT = a; }
        void set_max_trials(int a) { xcs.MAX_TRIALS = a; }
        void set_perf_trials(int a) { xcs.PERF_TRIALS = a; }
        void set_pop_max_size(int a) { xcs.POP_SIZE = a; }
        void set_loss_func(int a) { xcs.LOSS_FUNC = a; loss_set_func(&xcs); }
        void set_alpha(double a) { xcs.ALPHA = a; }
        void set_beta(double a) { xcs.BETA = a; }
        void set_delta(double a) { xcs.DELTA = a; }
        void set_eps_0(double a) { xcs.EPS_0 = a; } 
        void set_err_reduc(double a) { xcs.ERR_REDUC = a; }
        void set_fit_reduc(double a) { xcs.FIT_REDUC = a; }
        void set_init_error(double a) { xcs.INIT_ERROR = a; }
        void set_init_fitness(double a) { xcs.INIT_FITNESS = a; }
        void set_nu(double a) { xcs.NU = a; }
        void set_m_probation(int a) { xcs.M_PROBATION = a; }
        void set_theta_del(int a) { xcs.THETA_DEL = a; }
        void set_act_type(int a) { xcs.ACT_TYPE = a; }
        void set_cond_type(int a) { xcs.COND_TYPE = a; }
        void set_pred_type(int a) { xcs.PRED_TYPE = a; }
        void set_p_crossover(double a) { xcs.P_CROSSOVER = a; }
        void set_theta_ea(double a) { xcs.THETA_EA = a; }
        void set_lambda(int a) { xcs.LAMBDA = a; }
        void set_ea_select_type(int a) { xcs.EA_SELECT_TYPE = a; }
        void set_ea_select_size(double a) { xcs.EA_SELECT_SIZE = a; }
        void set_sam_type(int a) { xcs.SAM_TYPE = a; }
        void set_max_con(double a) { xcs.COND_MAX = a; }
        void set_min_con(double a) { xcs.COND_MIN = a; }
        void set_cond_smin(double a) { xcs.COND_SMIN = a; }
        void set_cond_bits(double a) { xcs.COND_BITS = a; }
        void set_cond_evolve_weights(_Bool a) { xcs.COND_EVOLVE_WEIGHTS = a; }
        void set_cond_evolve_neurons(_Bool a) { xcs.COND_EVOLVE_NEURONS = a; }
        void set_cond_evolve_functions(_Bool a) { xcs.COND_EVOLVE_FUNCTIONS = a; }
        void set_cond_output_activation(int a) { xcs.COND_OUTPUT_ACTIVATION = a; }
        void set_cond_hidden_activation(int a) { xcs.COND_HIDDEN_ACTIVATION = a; }
        void set_pred_output_activation(int a) { xcs.PRED_OUTPUT_ACTIVATION = a; }
        void set_pred_hidden_activation(int a) { xcs.PRED_HIDDEN_ACTIVATION = a; }
        void set_pred_momentum(double a) { xcs.PRED_MOMENTUM = a; }
        void set_pred_evolve_weights(_Bool a) { xcs.PRED_EVOLVE_WEIGHTS = a; }
        void set_pred_evolve_neurons(_Bool a) { xcs.PRED_EVOLVE_NEURONS = a; }
        void set_pred_evolve_functions(_Bool a) { xcs.PRED_EVOLVE_FUNCTIONS = a; }
        void set_pred_evolve_eta(_Bool a) { xcs.PRED_EVOLVE_ETA = a; }
        void set_pred_sgd_weights(_Bool a) { xcs.PRED_SGD_WEIGHTS = a; }
        void set_pred_reset(_Bool a) { xcs.PRED_RESET = a; }
        void set_max_neuron_mod(int a) { xcs.MAX_NEURON_MOD = a; }
        void set_dgp_num_nodes(int a) { xcs.DGP_NUM_NODES = a; }
        void set_reset_states(_Bool a) { xcs.RESET_STATES = a; }
        void set_max_k(int a) { xcs.MAX_K = a; }
        void set_max_t(int a) { xcs.MAX_T = a; }
        void set_gp_num_cons(int a) { xcs.GP_NUM_CONS = a; }
        void set_gp_init_depth(int a) { xcs.GP_INIT_DEPTH = a; }
        void set_pred_eta(double a) { xcs.PRED_ETA = a; }
        void set_cond_eta(double a) { xcs.COND_ETA = a; }
        void set_pred_x0(double a) { xcs.PRED_X0 = a; }
        void set_pred_rls_scale_factor(double a) { xcs.PRED_RLS_SCALE_FACTOR = a; }
        void set_pred_rls_lambda(double a) { xcs.PRED_RLS_LAMBDA = a; }
        void set_theta_sub(int a) { xcs.THETA_SUB = a; }
        void set_ea_subsumption(_Bool a) { xcs.EA_SUBSUMPTION = a; }
        void set_set_subsumption(_Bool a) { xcs.SET_SUBSUMPTION = a; }
        void set_teletransportation(int a) { xcs.TELETRANSPORTATION = a; }
        void set_gamma(double a) { xcs.GAMMA = a; }
        void set_p_explore(double a) { xcs.P_EXPLORE = a; }
};

PYBIND11_MODULE(xcsf, m)
{
    random_init();

    double (XCS::*fit1)(py::array_t<double>, py::array_t<double>, _Bool) = &XCS::fit;
    double (XCS::*fit2)(py::array_t<double>, py::array_t<double>,
            py::array_t<double>, py::array_t<double>, _Bool) = &XCS::fit;

    py::class_<XCS>(m, "XCS")
        .def(py::init<int, int>())
        .def(py::init<int, int, _Bool>())
        .def(py::init<int, int, const char *>())
        .def("fit", fit1)
        .def("fit", fit2)
        .def("predict", &XCS::predict)
        .def("score", &XCS::score)
        .def("save", &XCS::save)
        .def("load", &XCS::load)
        .def("version", &XCS::version)
        .def("single_update", &XCS::single_update)
        .def("single_error", &XCS::single_error)
        .def("single_decision", &XCS::single_decision)
        .def("single_reset", &XCS::single_reset)
        .def_property("OMP_NUM_THREADS", &XCS::get_omp_num_threads, &XCS::set_omp_num_threads)
        .def_property("POP_INIT", &XCS::get_pop_init, &XCS::set_pop_init)
        .def_property("MAX_TRIALS", &XCS::get_max_trials, &XCS::set_max_trials)
        .def_property("PERF_TRIALS", &XCS::get_perf_trials, &XCS::set_perf_trials)
        .def_property("POP_SIZE", &XCS::get_pop_max_size, &XCS::set_pop_max_size)
        .def_property("LOSS_FUNC", &XCS::get_loss_func, &XCS::set_loss_func)
        .def_property("ALPHA", &XCS::get_alpha, &XCS::set_alpha)
        .def_property("BETA", &XCS::get_beta, &XCS::set_beta)
        .def_property("DELTA", &XCS::get_delta, &XCS::set_delta)
        .def_property("EPS_0", &XCS::get_eps_0, &XCS::set_eps_0)
        .def_property("ERR_REDUC", &XCS::get_err_reduc, &XCS::set_err_reduc)
        .def_property("FIT_REDUC", &XCS::get_fit_reduc, &XCS::set_fit_reduc)
        .def_property("INIT_ERROR", &XCS::get_init_error, &XCS::set_init_error)
        .def_property("INIT_FITNESS", &XCS::get_init_fitness, &XCS::set_init_fitness)
        .def_property("NU", &XCS::get_nu, &XCS::set_nu)
        .def_property("M_PROBATION", &XCS::get_m_probation, &XCS::set_m_probation)
        .def_property("THETA_DEL", &XCS::get_theta_del, &XCS::set_theta_del)
        .def_property("ACT_TYPE", &XCS::get_act_type, &XCS::set_act_type)
        .def_property("COND_TYPE", &XCS::get_cond_type, &XCS::set_cond_type)
        .def_property("PRED_TYPE", &XCS::get_pred_type, &XCS::set_pred_type)
        .def_property("P_CROSSOVER", &XCS::get_p_crossover, &XCS::set_p_crossover)
        .def_property("THETA_EA", &XCS::get_theta_ea, &XCS::set_theta_ea)
        .def_property("LAMBDA", &XCS::get_lambda, &XCS::set_lambda)
        .def_property("EA_SELECT_TYPE", &XCS::get_ea_select_type, &XCS::set_ea_select_type)
        .def_property("EA_SELECT_SIZE", &XCS::get_ea_select_size, &XCS::set_ea_select_size)
        .def_property("SAM_TYPE", &XCS::get_sam_type, &XCS::set_sam_type)
        .def_property("COND_MAX", &XCS::get_max_con, &XCS::set_max_con)
        .def_property("COND_MIN", &XCS::get_min_con, &XCS::set_min_con)
        .def_property("COND_SMIN", &XCS::get_cond_smin, &XCS::set_cond_smin)
        .def_property("COND_BITS", &XCS::get_cond_bits, &XCS::set_cond_bits)
        .def_property("COND_EVOLVE_WEIGHTS", &XCS::get_cond_evolve_weights, &XCS::set_cond_evolve_weights)
        .def_property("COND_EVOLVE_NEURONS", &XCS::get_cond_evolve_neurons, &XCS::set_cond_evolve_neurons)
        .def_property("COND_EVOLVE_FUNCTIONS", &XCS::get_cond_evolve_functions, &XCS::set_cond_evolve_functions)
        .def_property("COND_NUM_NEURONS", &XCS::get_cond_num_neurons, &XCS::set_cond_num_neurons)
        .def_property("COND_MAX_NEURONS", &XCS::get_cond_max_neurons, &XCS::set_cond_max_neurons)
        .def_property("COND_OUTPUT_ACTIVATION", &XCS::get_cond_output_activation, &XCS::set_cond_output_activation)
        .def_property("COND_HIDDEN_ACTIVATION", &XCS::get_cond_hidden_activation, &XCS::set_cond_hidden_activation)
        .def_property("PRED_NUM_NEURONS", &XCS::get_pred_num_neurons, &XCS::set_pred_num_neurons)
        .def_property("PRED_MAX_NEURONS", &XCS::get_pred_max_neurons, &XCS::set_pred_max_neurons)
        .def_property("PRED_OUTPUT_ACTIVATION", &XCS::get_pred_output_activation, &XCS::set_pred_output_activation)
        .def_property("PRED_HIDDEN_ACTIVATION", &XCS::get_pred_hidden_activation, &XCS::set_pred_hidden_activation)
        .def_property("PRED_MOMENTUM", &XCS::get_pred_momentum, &XCS::set_pred_momentum)
        .def_property("PRED_EVOLVE_WEIGHTS", &XCS::get_pred_evolve_weights, &XCS::set_pred_evolve_weights)
        .def_property("PRED_EVOLVE_NEURONS", &XCS::get_pred_evolve_neurons, &XCS::set_pred_evolve_neurons)
        .def_property("PRED_EVOLVE_FUNCTIONS", &XCS::get_pred_evolve_functions, &XCS::set_pred_evolve_functions)
        .def_property("PRED_EVOLVE_ETA", &XCS::get_pred_evolve_eta, &XCS::set_pred_evolve_eta)
        .def_property("PRED_SGD_WEIGHTS", &XCS::get_pred_sgd_weights, &XCS::set_pred_sgd_weights)
        .def_property("PRED_RESET", &XCS::get_pred_reset, &XCS::set_pred_reset)
        .def_property("MAX_NEURON_MOD", &XCS::get_max_neuron_mod, &XCS::set_max_neuron_mod)
        .def_property("DGP_NUM_NODES", &XCS::get_dgp_num_nodes, &XCS::set_dgp_num_nodes)
        .def_property("RESET_STATES", &XCS::get_reset_states, &XCS::set_reset_states)
        .def_property("MAX_K", &XCS::get_max_k, &XCS::set_max_k)
        .def_property("MAX_T", &XCS::get_max_t, &XCS::set_max_t)
        .def_property("GP_NUM_CONS", &XCS::get_gp_num_cons, &XCS::set_gp_num_cons)
        .def_property("GP_INIT_DEPTH", &XCS::get_gp_init_depth, &XCS::set_gp_init_depth)
        .def_property("COND_ETA", &XCS::get_cond_eta, &XCS::set_cond_eta)
        .def_property("PRED_ETA", &XCS::get_pred_eta, &XCS::set_pred_eta)
        .def_property("PRED_X0", &XCS::get_pred_x0, &XCS::set_pred_x0)
        .def_property("PRED_RLS_SCALE_FACTOR", &XCS::get_pred_rls_scale_factor, &XCS::set_pred_rls_scale_factor)
        .def_property("PRED_RLS_LAMBDA", &XCS::get_pred_rls_lambda, &XCS::set_pred_rls_lambda)
        .def_property("THETA_SUB", &XCS::get_theta_sub, &XCS::set_theta_sub)
        .def_property("EA_SUBSUMPTION", &XCS::get_ea_subsumption, &XCS::set_ea_subsumption)
        .def_property("SET_SUBSUMPTION", &XCS::get_set_subsumption, &XCS::set_set_subsumption)
        .def_property("TELETRANSPORTATION", &XCS::get_teletransportation, &XCS::set_teletransportation)
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
        .def("print_pop", &XCS::print_pop)
        .def("msetsize", &XCS::get_msetsize)
        .def("mfrac", &XCS::get_mfrac)
        .def("print_params", &XCS::print_params)
        ;
}
