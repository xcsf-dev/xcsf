#include <string>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

extern "C" {   
	#include <stdbool.h>
	#include "data_structures.h"
	#include "cons.h"
	#include "random.h"
	#include "input.h"
	#include "cl_set.h"
}

extern "C" void experiment(XCSF *, INPUT *, INPUT *); 
 
/* flatten and convert numpy arrays */
void flatten(np::ndarray &orig, double *ret)
{
	int rows = orig.shape(0);
	int cols = orig.shape(1);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			ret[i*cols+j] = p::extract<double>(orig[i][j]);
		}
	}
}
 
/* XCSF class */
struct XCS
{        
	XCSF xcs;
	INPUT train_data;
	INPUT test_data;

	XCS(int num_x_vars, int num_y_vars) {
		random_init();
		constants_init(&xcs);
		xcs.num_x_vars = num_x_vars;
		xcs.num_y_vars = num_y_vars;
		pop_init(&xcs);

		train_data.rows = 0;
		train_data.x_cols = 0;
		train_data.y_cols = 0;
		test_data.rows = 0;
		test_data.x_cols = 0;
		test_data.y_cols = 0;
	}
 
	void fit() {
		experiment(&xcs, &train_data, &test_data);
	}

	void fit(np::ndarray &train_X,
			np::ndarray &train_Y, 
			np::ndarray &test_X,
			np::ndarray &test_Y) {

		// check inputs are correctly sized
		if(train_X.shape(0) != train_Y.shape(0)) {
			printf("error: training X and Y rows are not equal\n");
			return;
		}
		if(test_X.shape(0) != test_Y.shape(0)) {
			printf("error: testing X and Y rows are not equal\n");
			return;
		}
		if(train_X.shape(1) != test_X.shape(1)) {
			printf("error: number of training and testing X cols are not equal\n");
			return;
		}
		if(train_Y.shape(1) != test_Y.shape(1)) {
			printf("error: number of training and testing Y cols are not equal\n");
			return;
		}

		// clear any previous training data
		if(train_data.rows != 0) {
			free(train_data.x);
			free(train_data.y);
		}
		// load training data
		train_data.rows = train_X.shape(0);
		train_data.x_cols = train_X.shape(1);
		train_data.y_cols = train_Y.shape(1);
		train_data.x = (double *) malloc(sizeof(double) * train_data.rows * train_data.x_cols);
		train_data.y = (double *) malloc(sizeof(double) * train_data.rows * train_data.y_cols);
		flatten(train_X, train_data.x);
		flatten(train_Y, train_data.y);
 
		// clear any previous testing data
		if(test_data.rows != 0) {
			free(test_data.x);
			free(test_data.y);
		} 
		// load testing data
		test_data.rows = test_X.shape(0);
		test_data.x_cols = test_X.shape(1);
		test_data.y_cols = test_Y.shape(1);
		test_data.x = (double *) malloc(sizeof(double) * test_data.rows * test_data.x_cols);
		test_data.y = (double *) malloc(sizeof(double) * test_data.rows * test_data.y_cols);
		flatten(test_X, test_data.x);
		flatten(test_Y, test_data.y);

		// execute
		experiment(&xcs, &train_data, &test_data);
	}

	/* GETTERS */
	_Bool get_pop_init() {
		return xcs.POP_INIT;
	}

	double get_theta_mna() {
		return xcs.THETA_MNA;
	}

	int get_max_trials() {
		return xcs.MAX_TRIALS;
	}

	int get_perf_avg_trials() {
		return xcs.PERF_AVG_TRIALS;
	}

	int get_pop_size() {
		return xcs.POP_SIZE;
	}

	double get_alpha() {
		return xcs.ALPHA;
	}

	double get_beta() {
		return xcs.BETA;
	}

	double get_delta() {
		return xcs.DELTA;
	}

	double get_eps_0() {
		return xcs.EPS_0;
	} 

	double get_err_reduc() {
		return xcs.ERR_REDUC;
	}

	double get_fit_reduc() {
		return xcs.FIT_REDUC;
	}

	double get_init_error() {
		return xcs.INIT_ERROR;
	}

	double get_init_fitness() {
		return xcs.INIT_FITNESS;
	}

	double get_nu() {
		return xcs.NU;
	}

	double get_theta_del() {
		return xcs.THETA_DEL;
	}

	int get_cond_type() {
		return xcs.COND_TYPE;
	}

	int get_pred_type() {
		return xcs.PRED_TYPE;
	}

	double get_p_crossover() {
		return xcs.P_CROSSOVER;
	}

	double get_p_mutation() {
		return xcs.P_MUTATION;
	}

	double get_theta_ga() {
		return xcs.THETA_GA;
	}

	int get_theta_offspring() {
		return xcs.THETA_OFFSPRING;
	}

	double get_mueps_0() {
		return xcs.muEPS_0;
	}

	int get_num_sam() {
		return xcs.NUM_SAM;
	}

	double get_max_con() {
		return xcs.MAX_CON;
	}

	double get_min_con() {
		return xcs.MIN_CON;
	}

	double get_s_mutation() {
		return xcs.S_MUTATION;
	}

	int get_num_hidden_neurons() {
		return xcs.NUM_HIDDEN_NEURONS;
	}

	int get_dgp_num_nodes() {
		return xcs.DGP_NUM_NODES;
	}

	int get_gp_num_cons() {
		return xcs.GP_NUM_CONS;
	}

	double get_xcsf_eta() {
		return xcs.XCSF_ETA;
	}

	double get_xcsf_x0() {
		return xcs.XCSF_X0;
	}

	double get_theta_sub() {
		return xcs.THETA_SUB;
	}

	_Bool get_ga_subsumption() {
		return xcs.GA_SUBSUMPTION;
	}

	_Bool get_set_subsumption() {
		return xcs.SET_SUBSUMPTION;
	}

	int get_pop_num() {
		return xcs.pop_num;
	}

	int get_pop_num_sum() {
		return xcs.pop_num_sum;
	}

	double get_num_x_vars() {
		return xcs.num_x_vars;
	}

	double get_num_y_vars() {
		return xcs.num_y_vars;
	}                      

	/* SETTERS */
	void set_pop_init(_Bool a) {
		xcs.POP_INIT = a;
	}

	void set_theta_mna(double a) {
		xcs.THETA_MNA = a;
	}

	void set_max_trials(int a) {
		xcs.MAX_TRIALS = a;
	}

	void set_perf_avg_trials(int a) {
		xcs.PERF_AVG_TRIALS = a;
	}

	void set_pop_size(int a) {
		xcs.POP_SIZE = a;
	}

	void set_alpha(double a) {
		xcs.ALPHA = a;
	}

	void set_beta(double a) {
		xcs.BETA = a;
	}

	void set_delta(double a) {
		xcs.DELTA = a;
	}

	void set_eps_0(double a) {
		xcs.EPS_0 = a;
	} 

	void set_err_reduc(double a) {
		xcs.ERR_REDUC = a;
	}

	void set_fit_reduc(double a) {
		xcs.FIT_REDUC = a;
	}

	void set_init_error(double a) {
		xcs.INIT_ERROR = a;
	}

	void set_init_fitness(double a) {
		xcs.INIT_FITNESS = a;
	}

	void set_nu(double a) {
		xcs.NU = a;
	}

	void set_theta_del(double a) {
		xcs.THETA_DEL = a;
	}

	void set_cond_type(int a) {
		xcs.COND_TYPE = a;
	}

	void set_pred_type(int a) {
		xcs.PRED_TYPE = a;
	}

	void set_p_crossover(double a) {
		xcs.P_CROSSOVER = a;
	}

	void set_p_mutation(double a) {
		xcs.P_MUTATION = a;
	}

	void set_theta_ga(double a) {
		xcs.THETA_GA = a;
	}

	void set_theta_offspring(int a) {
		xcs.THETA_OFFSPRING = a;
	}

	void set_mueps_0(double a) {
		xcs.muEPS_0 = a;
	}

	void set_num_sam(int a) {
		xcs.NUM_SAM = a;
	}

	void set_max_con(double a) {
		xcs.MAX_CON = a;
	}

	void set_min_con(double a) {
		xcs.MIN_CON = a;
	}

	void set_s_mutation(double a) {
		xcs.S_MUTATION = a;
	}

	void set_num_hidden_neurons(int a) {
		xcs.NUM_HIDDEN_NEURONS = a;
	}

	void set_dgp_num_nodes(int a) {
		xcs.DGP_NUM_NODES = a;
	}

	void set_gp_num_cons(int a) {
		xcs.GP_NUM_CONS = a;
	}

	void set_xcsf_eta(double a) {
		xcs.XCSF_ETA = a;
	}

	void set_xcsf_x0(double a) {
		xcs.XCSF_X0 = a;
	}

	void set_theta_sub(double a) {
		xcs.THETA_SUB = a;
	}

	void set_ga_subsumption(_Bool a) {
		xcs.GA_SUBSUMPTION = a;
	}

	void set_set_subsumption(_Bool a) {
		xcs.SET_SUBSUMPTION = a;
	}
};

BOOST_PYTHON_MODULE(xcsf)
{
	np::initialize();

	void (XCS::*fit1)() = &XCS::fit;
	void (XCS::*fit2)(np::ndarray&, np::ndarray&, np::ndarray&, np::ndarray&) = &XCS::fit;

	p::class_<XCS>("XCS", p::init<int, int>())
		.def("fit", fit1)
		.def("fit", fit2)
		.def("get_pop_init", &XCS::get_pop_init)
		.def("get_theta_mna", &XCS::get_theta_mna)
		.def("get_max_trials", &XCS::get_max_trials)
		.def("get_perf_avg_trials", &XCS::get_perf_avg_trials)
		.def("get_pop_size", &XCS::get_pop_size)
		.def("get_alpha", &XCS::get_alpha)
		.def("get_beta", &XCS::get_beta)
		.def("get_delta", &XCS::get_delta)
		.def("get_eps_0", &XCS::get_eps_0)
		.def("get_err_reduc", &XCS::get_err_reduc)
		.def("get_fit_reduc", &XCS::get_fit_reduc)
		.def("get_init_error", &XCS::get_init_error)
		.def("get_init_fitness", &XCS::get_init_fitness)
		.def("get_nu", &XCS::get_nu)
		.def("get_theta_del", &XCS::get_theta_del)
		.def("get_cond_type", &XCS::get_cond_type)
		.def("get_pred_type", &XCS::get_pred_type)
		.def("get_p_crossover", &XCS::get_p_crossover)
		.def("get_p_mutation", &XCS::get_p_mutation)
		.def("get_theta_ga", &XCS::get_theta_ga)
		.def("get_theta_offspring", &XCS::get_theta_offspring)
		.def("get_mueps_0", &XCS::get_mueps_0)
		.def("get_num_sam", &XCS::get_num_sam)
		.def("get_max_con", &XCS::get_max_con)
		.def("get_min_con", &XCS::get_min_con)
		.def("get_s_mutation", &XCS::get_s_mutation)
		.def("get_num_hidden_neurons", &XCS::get_num_hidden_neurons)
		.def("get_dgp_num_nodes", &XCS::get_dgp_num_nodes)
		.def("get_gp_num_cons", &XCS::get_gp_num_cons)
		.def("get_xcsf_eta", &XCS::get_xcsf_eta)
		.def("get_xcsf_x0", &XCS::get_xcsf_x0)
		.def("get_theta_sub", &XCS::get_theta_sub)
		.def("get_ga_subsumption", &XCS::get_ga_subsumption)
		.def("get_set_subsumption", &XCS::get_set_subsumption)
		.def("get_pop_num", &XCS::get_pop_num)
		.def("get_pop_num_sum", &XCS::get_pop_num_sum)
		.def("get_num_x_vars", &XCS::get_num_x_vars)
		.def("get_num_y_vars", &XCS::get_num_x_vars)
		.def("set_pop_init", &XCS::set_pop_init)
		.def("set_theta_mna", &XCS::set_theta_mna)
		.def("set_max_trials", &XCS::set_max_trials)
		.def("set_perf_avg_trials", &XCS::set_perf_avg_trials)
		.def("set_pop_size", &XCS::set_pop_size)
		.def("set_alpha", &XCS::set_alpha)
		.def("set_beta", &XCS::set_beta)
		.def("set_delta", &XCS::set_delta)
		.def("set_eps_0", &XCS::set_eps_0)
		.def("set_err_reduc", &XCS::set_err_reduc)
		.def("set_fit_reduc", &XCS::set_fit_reduc)
		.def("set_init_error", &XCS::set_init_error)
		.def("set_init_fitness", &XCS::set_init_fitness)
		.def("set_nu", &XCS::set_nu)
		.def("set_theta_del", &XCS::set_theta_del)
		.def("set_cond_type", &XCS::set_cond_type)
		.def("set_pred_type", &XCS::set_pred_type)
		.def("set_p_crossover", &XCS::set_p_crossover)
		.def("set_p_mutation", &XCS::set_p_mutation)
		.def("set_theta_ga", &XCS::set_theta_ga)
		.def("set_theta_offspring", &XCS::set_theta_offspring)
		.def("set_mueps_0", &XCS::set_mueps_0)
		.def("set_num_sam", &XCS::set_num_sam)
		.def("set_max_con", &XCS::set_max_con)
		.def("set_min_con", &XCS::set_min_con)
		.def("set_s_mutation", &XCS::set_s_mutation)
		.def("set_num_hidden_neurons", &XCS::set_num_hidden_neurons)
		.def("set_dgp_num_nodes", &XCS::set_dgp_num_nodes)
		.def("set_gp_num_cons", &XCS::set_gp_num_cons)
		.def("set_xcsf_eta", &XCS::set_xcsf_eta)
		.def("set_xcsf_x0", &XCS::set_xcsf_x0)
		.def("set_theta_sub", &XCS::set_theta_sub)
		.def("set_ga_subsumption", &XCS::set_ga_subsumption)
		.def("set_set_subsumption", &XCS::set_set_subsumption)
		;
}
