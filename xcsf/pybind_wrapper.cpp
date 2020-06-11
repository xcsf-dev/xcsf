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

#include "../lib/pybind11/include/pybind11/pybind11.h"
#include "../lib/pybind11/include/pybind11/numpy.h"
#include <string>
#include <vector>

namespace py = pybind11;

extern "C" {
#include <stdbool.h>
#include "xcsf.h"
#include "xcs_rl.h"
#include "xcs_supervised.h"
#include "pa.h"
#include "config.h"
#include "param.h"
#include "utils.h"
#include "clset.h"
}

/**
 * @brief Python XCSF class data structure.
 */
class XCS
{
    private:
        XCSF xcs; //!< XCSF data structure
        double *state; //!< Current input state for RL
        int action; //!< Current action for RL
        double payoff; //!< Current reward for RL
        INPUT *train_data; //!< Current training data for supervised learning
        INPUT *test_data; //!< Currrent test data for supervised learning

    public:
        /**
         * @brief Constructor with default config.
         */
        XCS(int x_dim, int y_dim, int n_actions) :
            XCS(x_dim, y_dim, n_actions, "default.ini") {}

        /**
         * @brief Constructor with a specified config.
         */
        XCS(int x_dim, int y_dim, int n_actions, const char *filename)
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
            train_data = (INPUT *)malloc(sizeof(INPUT));
            train_data->n_samples = 0;
            train_data->x_dim = 0;
            train_data->y_dim = 0;
            train_data->x = NULL;
            train_data->y = NULL;
            test_data = (INPUT *)malloc(sizeof(INPUT));
            test_data->n_samples = 0;
            test_data->x_dim = 0;
            test_data->y_dim = 0;
            test_data->x = NULL;
            test_data->y = NULL;
        }

        int version_major()
        {
            return VERSION_MAJOR;
        }

        int version_minor()
        {
            return VERSION_MINOR;
        }

        int version_build()
        {
            return VERSION_BUILD;
        }

        size_t save(char *fname)
        {
            return xcsf_save(&xcs, fname);
        }

        size_t load(char *fname)
        {
            return xcsf_load(&xcs, fname);
        }

        void print_params()
        {
            param_print(&xcs);
        }

        void ae_expand()
        {
            xcsf_ae_expand(&xcs);
        }

        void ae_to_classifier(int y_dim)
        {
            xcsf_ae_to_classifier(&xcs, y_dim);
        }

        void print_pop(_Bool printc, _Bool printa, _Bool printp)
        {
            xcsf_print_pop(&xcs, printc, printa, printp);
        }

        /* Reinforcement learning */

        void init_trial()
        {
            if(xcs.time == 0) {
                clset_pop_init(&xcs);
            }
            xcs_rl_init_trial(&xcs);
        }

        void end_trial()
        {
            xcs_rl_end_trial(&xcs);
        }

        void init_step()
        {
            xcs_rl_init_step(&xcs);
        }

        void end_step()
        {
            xcs_rl_end_step(&xcs, state, action, payoff);
        }

        int decision(py::array_t<double> input, _Bool explore)
        {
            py::buffer_info buf = input.request();
            state = (double *) buf.ptr;
            param_set_explore(&xcs, explore);
            action = xcs_rl_decision(&xcs, state);
            return action;
        }

        void update(double reward, _Bool reset)
        {
            payoff = reward;
            xcs_rl_update(&xcs, state, action, payoff, reset);
        }

        double error(double reward, _Bool reset, double max_p)
        {
            payoff = reward;
            return xcs_rl_error(&xcs, action, payoff, reset, max_p);
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
            return xcs_supervised_fit(&xcs, train_data, NULL, shuffle);
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
            return xcs_supervised_fit(&xcs, train_data, test_data, shuffle);
        }

        py::array_t<double> predict(py::array_t<double> x)
        {
            // inputs to predict
            py::buffer_info buf_x = x.request();
            int n_samples = buf_x.shape[0];
            double *input = (double *) buf_x.ptr;
            // predicted outputs
            double *output = (double *) malloc(sizeof(double) * n_samples * xcs.y_dim);
            xcs_supervised_predict(&xcs, input, output, n_samples);
            // return numpy array
            return py::array_t<double>(std::vector<ptrdiff_t> {n_samples, xcs.y_dim}, output);
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
            return xcs_supervised_score(&xcs, test_data);
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

        int get_omp_num_threads()
        {
            return xcs.OMP_NUM_THREADS;
        }

        _Bool get_pop_init()
        {
            return xcs.POP_INIT;
        }

        _Bool get_auto_encode()
        {
            return xcs.AUTO_ENCODE;
        }

        int get_max_trials()
        {
            return xcs.MAX_TRIALS;
        }

        int get_perf_trials()
        {
            return xcs.PERF_TRIALS;
        }

        int get_pop_max_size()
        {
            return xcs.POP_SIZE;
        }

        int get_loss_func()
        {
            return xcs.LOSS_FUNC;
        }

        double get_alpha()
        {
            return xcs.ALPHA;
        }

        double get_beta()
        {
            return xcs.BETA;
        }

        double get_delta()
        {
            return xcs.DELTA;
        }

        double get_eps_0()
        {
            return xcs.EPS_0;
        }

        double get_err_reduc()
        {
            return xcs.ERR_REDUC;
        }

        double get_fit_reduc()
        {
            return xcs.FIT_REDUC;
        }

        double get_init_error()
        {
            return xcs.INIT_ERROR;
        }

        double get_init_fitness()
        {
            return xcs.INIT_FITNESS;
        }

        double get_nu()
        {
            return xcs.NU;
        }

        int get_m_probation()
        {
            return xcs.M_PROBATION;
        }

        int get_theta_del()
        {
            return xcs.THETA_DEL;
        }

        int get_act_type()
        {
            return xcs.ACT_TYPE;
        }

        int get_cond_type()
        {
            return xcs.COND_TYPE;
        }

        int get_pred_type()
        {
            return xcs.PRED_TYPE;
        }

        double get_p_crossover()
        {
            return xcs.P_CROSSOVER;
        }

        double get_theta_ea()
        {
            return xcs.THETA_EA;
        }

        int get_lambda()
        {
            return xcs.LAMBDA;
        }

        int get_ea_select_type()
        {
            return xcs.EA_SELECT_TYPE;
        }

        double get_ea_select_size()
        {
            return xcs.EA_SELECT_SIZE;
        }

        int get_sam_type()
        {
            return xcs.SAM_TYPE;
        }

        double get_max_con()
        {
            return xcs.COND_MAX;
        }

        double get_min_con()
        {
            return xcs.COND_MIN;
        }

        double get_cond_smin()
        {
            return xcs.COND_SMIN;
        }

        int get_cond_bits()
        {
            return xcs.COND_BITS;
        }

        _Bool get_cond_evolve_weights()
        {
            return xcs.COND_EVOLVE_WEIGHTS;
        }

        _Bool get_cond_evolve_neurons()
        {
            return xcs.COND_EVOLVE_NEURONS;
        }

        _Bool get_cond_evolve_functions()
        {
            return xcs.COND_EVOLVE_FUNCTIONS;
        }

        _Bool get_cond_evolve_connectivity()
        {
            return xcs.COND_EVOLVE_CONNECTIVITY;
        }

        int get_cond_output_activation()
        {
            return xcs.COND_OUTPUT_ACTIVATION;
        }

        int get_cond_hidden_activation()
        {
            return xcs.COND_HIDDEN_ACTIVATION;
        }

        int get_pred_output_activation()
        {
            return xcs.PRED_OUTPUT_ACTIVATION;
        }

        int get_pred_hidden_activation()
        {
            return xcs.PRED_HIDDEN_ACTIVATION;
        }

        double get_pred_momentum()
        {
            return xcs.PRED_MOMENTUM;
        }

        _Bool get_pred_evolve_weights()
        {
            return xcs.PRED_EVOLVE_WEIGHTS;
        }

        _Bool get_pred_evolve_neurons()
        {
            return xcs.PRED_EVOLVE_NEURONS;
        }

        _Bool get_pred_evolve_functions()
        {
            return xcs.PRED_EVOLVE_FUNCTIONS;
        }

        _Bool get_pred_evolve_connectivity()
        {
            return xcs.PRED_EVOLVE_CONNECTIVITY;
        }

        _Bool get_pred_evolve_eta()
        {
            return xcs.PRED_EVOLVE_ETA;
        }

        _Bool get_pred_sgd_weights()
        {
            return xcs.PRED_SGD_WEIGHTS;
        }

        _Bool get_pred_reset()
        {
            return xcs.PRED_RESET;
        }

        int get_max_neuron_mod()
        {
            return xcs.MAX_NEURON_MOD;
        }

        int get_dgp_num_nodes()
        {
            return xcs.DGP_NUM_NODES;
        }

        _Bool get_reset_states()
        {
            return xcs.RESET_STATES;
        }

        int get_max_k()
        {
            return xcs.MAX_K;
        }

        int get_max_t()
        {
            return xcs.MAX_T;
        }

        int get_gp_num_cons()
        {
            return xcs.GP_NUM_CONS;
        }

        int get_gp_init_depth()
        {
            return xcs.GP_INIT_DEPTH;
        }

        double get_pred_eta()
        {
            return xcs.PRED_ETA;
        }

        double get_cond_eta()
        {
            return xcs.COND_ETA;
        }

        double get_pred_x0()
        {
            return xcs.PRED_X0;
        }

        double get_pred_rls_scale_factor()
        {
            return xcs.PRED_RLS_SCALE_FACTOR;
        }

        double get_pred_rls_lambda()
        {
            return xcs.PRED_RLS_LAMBDA;
        }

        int get_theta_sub()
        {
            return xcs.THETA_SUB;
        }

        _Bool get_ea_subsumption()
        {
            return xcs.EA_SUBSUMPTION;
        }

        _Bool get_set_subsumption()
        {
            return xcs.SET_SUBSUMPTION;
        }

        int get_pop_size()
        {
            return xcs.pset.size;
        }

        int get_pop_num()
        {
            return xcs.pset.num;
        }

        int get_time()
        {
            return xcs.time;
        }

        double get_x_dim()
        {
            return xcs.x_dim;
        }

        double get_y_dim()
        {
            return xcs.y_dim;
        }

        double get_n_actions()
        {
            return xcs.n_actions;
        }

        double get_pop_mean_cond_size()
        {
            return clset_mean_cond_size(&xcs, &xcs.pset);
        }

        double get_pop_mean_pred_size()
        {
            return clset_mean_pred_size(&xcs, &xcs.pset);
        }

        double get_pop_mean_pred_eta(int layer)
        {
            return clset_mean_eta(&xcs, &xcs.pset, layer);
        }

        double get_pop_mean_pred_neurons(int layer)
        {
            return clset_mean_neurons(&xcs, &xcs.pset, layer);
        }

        double get_pop_mean_pred_layers()
        {
            return clset_mean_layers(&xcs, &xcs.pset);
        }

        double get_msetsize()
        {
            return xcs.msetsize;
        }

        double get_mfrac()
        {
            return xcs.mfrac;
        }

        int get_teletransportation()
        {
            return xcs.TELETRANSPORTATION;
        }

        double get_gamma()
        {
            return xcs.GAMMA;
        }

        double get_p_explore()
        {
            return xcs.P_EXPLORE;
        }

        /* SETTERS */

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

        void set_omp_num_threads(int a)
        {
            param_set_omp_num_threads(&xcs, a);
        }

        void set_pop_init(_Bool a)
        {
            param_set_pop_init(&xcs, a);
        }

        void set_auto_encode(_Bool a)
        {
            param_set_auto_encode(&xcs, a);
        }

        void set_max_trials(int a)
        {
            param_set_max_trials(&xcs, a);
        }

        void set_perf_trials(int a)
        {
            param_set_perf_trials(&xcs, a);
        }

        void set_pop_max_size(int a)
        {
            param_set_pop_size(&xcs, a);
        }

        void set_loss_func(int a)
        {
            param_set_loss_func(&xcs, a);
        }

        void set_alpha(double a)
        {
            param_set_alpha(&xcs, a);
        }

        void set_beta(double a)
        {
            param_set_beta(&xcs, a);
        }

        void set_delta(double a)
        {
            param_set_delta(&xcs, a);
        }

        void set_eps_0(double a)
        {
            param_set_eps_0(&xcs, a);
        }

        void set_err_reduc(double a)
        {
            param_set_err_reduc(&xcs, a);
        }

        void set_fit_reduc(double a)
        {
            param_set_fit_reduc(&xcs, a);
        }

        void set_init_error(double a)
        {
            param_set_init_error(&xcs, a);
        }

        void set_init_fitness(double a)
        {
            param_set_init_fitness(&xcs, a);
        }

        void set_nu(double a)
        {
            param_set_nu(&xcs, a);
        }

        void set_m_probation(int a)
        {
            param_set_m_probation(&xcs, a);
        }

        void set_theta_del(int a)
        {
            param_set_theta_del(&xcs, a);
        }

        void set_act_type(int a)
        {
            param_set_act_type(&xcs, a);
        }

        void set_cond_type(int a)
        {
            param_set_cond_type(&xcs, a);
        }

        void set_pred_type(int a)
        {
            param_set_pred_type(&xcs, a);
        }

        void set_p_crossover(double a)
        {
            param_set_p_crossover(&xcs, a);
        }

        void set_theta_ea(double a)
        {
            param_set_theta_ea(&xcs, a);
        }

        void set_lambda(int a)
        {
            param_set_lambda(&xcs, a);
        }

        void set_ea_select_type(int a)
        {
            param_set_ea_select_type(&xcs, a);
        }

        void set_ea_select_size(double a)
        {
            param_set_ea_select_size(&xcs, a);
        }

        void set_sam_type(int a)
        {
            param_set_sam_type(&xcs, a);
        }

        void set_max_con(double a)
        {
            param_set_cond_max(&xcs, a);
        }

        void set_min_con(double a)
        {
            param_set_cond_min(&xcs, a);
        }

        void set_cond_smin(double a)
        {
            param_set_cond_smin(&xcs, a);
        }

        void set_cond_bits(int a)
        {
            param_set_cond_bits(&xcs, a);
        }

        void set_cond_evolve_weights(_Bool a)
        {
            param_set_cond_evolve_weights(&xcs, a);
        }

        void set_cond_evolve_neurons(_Bool a)
        {
            param_set_cond_evolve_neurons(&xcs, a);
        }

        void set_cond_evolve_functions(_Bool a)
        {
            param_set_cond_evolve_functions(&xcs, a);
        }

        void set_cond_evolve_connectivity(_Bool a)
        {
            param_set_cond_evolve_connectivity(&xcs, a);
        }

        void set_cond_output_activation(int a)
        {
            param_set_cond_output_activation(&xcs, a);
        }

        void set_cond_hidden_activation(int a)
        {
            param_set_cond_hidden_activation(&xcs, a);
        }

        void set_pred_output_activation(int a)
        {
            param_set_pred_output_activation(&xcs, a);
        }

        void set_pred_hidden_activation(int a)
        {
            param_set_pred_hidden_activation(&xcs, a);
        }

        void set_pred_momentum(double a)
        {
            param_set_pred_momentum(&xcs, a);
        }

        void set_pred_evolve_weights(_Bool a)
        {
            param_set_pred_evolve_weights(&xcs, a);
        }

        void set_pred_evolve_neurons(_Bool a)
        {
            param_set_pred_evolve_neurons(&xcs, a);
        }

        void set_pred_evolve_functions(_Bool a)
        {
            param_set_pred_evolve_functions(&xcs, a);
        }

        void set_pred_evolve_connectivity(_Bool a)
        {
            param_set_pred_evolve_connectivity(&xcs, a);
        }

        void set_pred_evolve_eta(_Bool a)
        {
            param_set_pred_evolve_eta(&xcs, a);
        }

        void set_pred_sgd_weights(_Bool a)
        {
            param_set_pred_sgd_weights(&xcs, a);
        }

        void set_pred_reset(_Bool a)
        {
            param_set_pred_reset(&xcs, a);
        }

        void set_max_neuron_mod(int a)
        {
            param_set_max_neuron_mod(&xcs, a);
        }

        void set_dgp_num_nodes(int a)
        {
            param_set_dgp_num_nodes(&xcs, a);
        }

        void set_reset_states(_Bool a)
        {
            param_set_reset_states(&xcs, a);
        }

        void set_max_k(int a)
        {
            param_set_max_k(&xcs, a);
        }

        void set_max_t(int a)
        {
            param_set_max_t(&xcs, a);
        }

        void set_gp_num_cons(int a)
        {
            param_set_gp_num_cons(&xcs, a);
        }

        void set_gp_init_depth(int a)
        {
            param_set_gp_init_depth(&xcs, a);
        }

        void set_pred_eta(double a)
        {
            param_set_pred_eta(&xcs, a);
        }

        void set_cond_eta(double a)
        {
            param_set_cond_eta(&xcs, a);
        }

        void set_pred_x0(double a)
        {
            param_set_pred_x0(&xcs, a);
        }

        void set_pred_rls_scale_factor(double a)
        {
            param_set_pred_rls_scale_factor(&xcs, a);
        }

        void set_pred_rls_lambda(double a)
        {
            param_set_pred_rls_lambda(&xcs, a);
        }

        void set_theta_sub(int a)
        {
            param_set_theta_sub(&xcs, a);
        }

        void set_ea_subsumption(_Bool a)
        {
            param_set_ea_subsumption(&xcs, a);
        }

        void set_set_subsumption(_Bool a)
        {
            param_set_set_subsumption(&xcs, a);
        }

        void set_teletransportation(int a)
        {
            param_set_teletransportation(&xcs, a);
        }

        void set_gamma(double a)
        {
            param_set_gamma(&xcs, a);
        }

        void set_p_explore(double a)
        {
            param_set_p_explore(&xcs, a);
        }
};

PYBIND11_MODULE(xcsf, m)
{
    random_init();
    double (XCS::*fit1)(py::array_t<double>, py::array_t<double>, _Bool) = &XCS::fit;
    double (XCS::*fit2)(py::array_t<double>, py::array_t<double>,
                        py::array_t<double>, py::array_t<double>, _Bool) = &XCS::fit;
    py::class_<XCS>(m, "XCS")
    .def(py::init<int, int, int>())
    .def(py::init<int, int, int, const char *>())
    .def("fit", fit1)
    .def("fit", fit2)
    .def("predict", &XCS::predict)
    .def("score", &XCS::score)
    .def("save", &XCS::save)
    .def("load", &XCS::load)
    .def("version_major", &XCS::version_major)
    .def("version_minor", &XCS::version_minor)
    .def("version_build", &XCS::version_build)
    .def("init_trial", &XCS::init_trial)
    .def("end_trial", &XCS::end_trial)
    .def("init_step", &XCS::init_step)
    .def("end_step", &XCS::end_step)
    .def("decision", &XCS::decision)
    .def("update", &XCS::update)
    .def("error", &XCS::error)
    .def_property("OMP_NUM_THREADS", &XCS::get_omp_num_threads, &XCS::set_omp_num_threads)
    .def_property("POP_INIT", &XCS::get_pop_init, &XCS::set_pop_init)
    .def_property("AUTO_ENCODE", &XCS::get_auto_encode, &XCS::set_auto_encode)
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
    .def_property("COND_EVOLVE_WEIGHTS", &XCS::get_cond_evolve_weights,
                  &XCS::set_cond_evolve_weights)
    .def_property("COND_EVOLVE_NEURONS", &XCS::get_cond_evolve_neurons,
                  &XCS::set_cond_evolve_neurons)
    .def_property("COND_EVOLVE_FUNCTIONS", &XCS::get_cond_evolve_functions,
                  &XCS::set_cond_evolve_functions)
    .def_property("COND_EVOLVE_CONNECTIVITY", &XCS::get_cond_evolve_connectivity,
                  &XCS::set_cond_evolve_connectivity)
    .def_property("COND_NUM_NEURONS", &XCS::get_cond_num_neurons, &XCS::set_cond_num_neurons)
    .def_property("COND_MAX_NEURONS", &XCS::get_cond_max_neurons, &XCS::set_cond_max_neurons)
    .def_property("COND_OUTPUT_ACTIVATION", &XCS::get_cond_output_activation,
                  &XCS::set_cond_output_activation)
    .def_property("COND_HIDDEN_ACTIVATION", &XCS::get_cond_hidden_activation,
                  &XCS::set_cond_hidden_activation)
    .def_property("PRED_NUM_NEURONS", &XCS::get_pred_num_neurons, &XCS::set_pred_num_neurons)
    .def_property("PRED_MAX_NEURONS", &XCS::get_pred_max_neurons, &XCS::set_pred_max_neurons)
    .def_property("PRED_OUTPUT_ACTIVATION", &XCS::get_pred_output_activation,
                  &XCS::set_pred_output_activation)
    .def_property("PRED_HIDDEN_ACTIVATION", &XCS::get_pred_hidden_activation,
                  &XCS::set_pred_hidden_activation)
    .def_property("PRED_MOMENTUM", &XCS::get_pred_momentum, &XCS::set_pred_momentum)
    .def_property("PRED_EVOLVE_WEIGHTS", &XCS::get_pred_evolve_weights,
                  &XCS::set_pred_evolve_weights)
    .def_property("PRED_EVOLVE_NEURONS", &XCS::get_pred_evolve_neurons,
                  &XCS::set_pred_evolve_neurons)
    .def_property("PRED_EVOLVE_FUNCTIONS", &XCS::get_pred_evolve_functions,
                  &XCS::set_pred_evolve_functions)
    .def_property("PRED_EVOLVE_CONNECTIVITY", &XCS::get_pred_evolve_connectivity,
                  &XCS::set_pred_evolve_connectivity)
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
    .def_property("PRED_RLS_SCALE_FACTOR", &XCS::get_pred_rls_scale_factor,
                  &XCS::set_pred_rls_scale_factor)
    .def_property("PRED_RLS_LAMBDA", &XCS::get_pred_rls_lambda, &XCS::set_pred_rls_lambda)
    .def_property("THETA_SUB", &XCS::get_theta_sub, &XCS::set_theta_sub)
    .def_property("EA_SUBSUMPTION", &XCS::get_ea_subsumption, &XCS::set_ea_subsumption)
    .def_property("SET_SUBSUMPTION", &XCS::get_set_subsumption, &XCS::set_set_subsumption)
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
    .def("print_pop", &XCS::print_pop)
    .def("msetsize", &XCS::get_msetsize)
    .def("mfrac", &XCS::get_mfrac)
    .def("print_params", &XCS::print_params)
    .def("ae_expand", &XCS::ae_expand)
    .def("ae_to_classifier", &XCS::ae_to_classifier)
    ;
}
