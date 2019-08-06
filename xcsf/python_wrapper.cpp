#include <string>

extern "C" {   
	#include <stdbool.h>
	#include "data_structures.h"
	#include "cons.h"
	#include "random.h"
	#include "input.h"
	#include "cl_set.h"
}

extern "C" void experiment(XCSF *, INPUT *, INPUT *); 

struct XCS
{        
	XCSF xcs;
	INPUT train_data;
	INPUT test_data;

	XCS(char *infname, int max_trials) {
		random_init();
		constants_init(&xcs);
		xcs.MAX_TRIALS = max_trials;
		input_read_csv(infname, &train_data, &test_data);
		xcs.num_x_vars = train_data.x_cols;
		xcs.num_y_vars = train_data.y_cols;
		pop_init(&xcs);
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
     
	//double *gp_cons;
                
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
 
	void fit() {
		experiment(&xcs, &train_data, &test_data);
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
};

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(xcsf)
{
	class_<XCS>("XCS", init<char *, int>())
		.def("fit", &XCS::fit)
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
		;                         
}
