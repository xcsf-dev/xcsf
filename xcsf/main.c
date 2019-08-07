/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
 *
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
 *
 **************
 * Description: 
 **************
 * The main XCSF module.  
 *
 * 1) Initialises the environment: constants, random number generator, problem
 * function, and performance output writing.
 *
 * 2) Executes the experiments: iteratively retrieving a problem instance,
 * generating a match set, calculating a system prediction, providing
 * reinforcement and running the genetic algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include "data_structures.h"
#include "cons.h"
#include "random.h"
#include "cl.h"
#include "cl_set.h"
#include "ga.h"
#include "input.h"
#include "perf.h"

void experiment1(XCSF *xcsf, INPUT *train_data);
void experiment2(XCSF *xcsf, INPUT *train_data, INPUT *test_data);
void trial(XCSF *xcsf, int cnt, double *x, double *y, _Bool train, double *err);

int main(int argc, char **argv)
{    
	if(argc < 2 || argc > 3) {
		printf("Usage: xcsf inputfile [MaxTrials]\n");
		exit(EXIT_FAILURE);
	} 

	random_init();

	// initialise XCSF
	XCSF *xcsf = malloc(sizeof(XCSF));
	constants_init(xcsf); // read cons.txt default parameters      
	// override with command line values
	if(argc > 2) {
		xcsf->MAX_TRIALS = atoi(argv[2]);
	}    

	// read csv input data
	INPUT *train_data = malloc(sizeof(INPUT));
	INPUT *test_data = malloc(sizeof(INPUT));
	input_read_csv(argv[1], train_data, test_data);

	xcsf->num_x_vars = train_data->x_cols;
	xcsf->num_y_vars = train_data->y_cols;

	// initialise population
	pop_init(xcsf);
	// run an experiment
	experiment2(xcsf, train_data, test_data);

	// clean up
	set_kill(xcsf, &xcsf->pset);
	constants_free(xcsf);        
	free(xcsf);
	input_free(train_data);
	input_free(test_data);
	free(train_data);
	free(test_data);

	return EXIT_SUCCESS;
}

void experiment1(XCSF *xcsf, INPUT *train_data)
{  
#ifdef GNUPLOT
	gplot_init(xcsf);
#endif
 
	// performance tracking
	double err[xcsf->PERF_AVG_TRIALS];

	// current input
	double *x = malloc(sizeof(double)*xcsf->num_x_vars);
	double *y = malloc(sizeof(double)*xcsf->num_y_vars);
 
	// each trial in an experiment
	for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
		// train
		input_rand_sample(train_data, x, y);
		trial(xcsf, cnt, x, y, true, err);
		// display performance
		if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
			disp_perf1(xcsf, err, cnt);
		}
	}

	// clean up
	free(x);
	free(y);      

#ifdef GNUPLOT
	gplot_free(xcsf);
#endif
}

void experiment2(XCSF *xcsf, INPUT *train_data, INPUT *test_data)
{   
#ifdef GNUPLOT
	gplot_init(xcsf);
#endif
 
	// performance tracking
	double err[xcsf->PERF_AVG_TRIALS];
	double terr[xcsf->PERF_AVG_TRIALS];

	// current input
	double *x = malloc(sizeof(double)*xcsf->num_x_vars);
	double *y = malloc(sizeof(double)*xcsf->num_y_vars);

	// each trial in an experiment
	for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
		// train
		input_rand_sample(train_data, x, y);
		trial(xcsf, cnt, x, y, true, err);
		// test
		input_rand_sample(test_data, x, y);
		trial(xcsf, cnt, x, y, false, terr);
		// display performance
		if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
			disp_perf2(xcsf, err, terr, cnt);
		}
	}

	// clean up
	free(x);
	free(y);

#ifdef GNUPLOT
	gplot_free(xcsf);
#endif
}

void trial(XCSF *xcsf, int cnt, double *x, double *y, _Bool train, double *err)
{
	// create match set
	NODE *mset = NULL, *kset = NULL;
	int msize = 0, mnum = 0;
	set_match(xcsf, &mset, &msize, &mnum, x, cnt, &kset);

	// calculate system prediction and track performance
	double *pred = malloc(sizeof(double)*xcsf->num_y_vars);
	set_pred(xcsf, &mset, msize, x, pred);
	err[cnt % xcsf->PERF_AVG_TRIALS] = 0.0;
	for(int i = 0; i < xcsf->num_y_vars; i++) {
		err[cnt % xcsf->PERF_AVG_TRIALS] += (y[i]-pred[i])*(y[i]-pred[i]);
	}
	err[cnt % xcsf->PERF_AVG_TRIALS] /= (double)xcsf->num_y_vars; // MSE

	if(train) {
		// provide reinforcement to the set
		set_update(xcsf, &mset, &msize, &mnum, y, &kset, x);
		// run the genetic algorithm
		ga(xcsf, &mset, msize, mnum, cnt, &kset);
	}

	// clean up
	free(pred);
	set_kill(xcsf, &kset); // kills deleted classifiers
	set_free(xcsf, &mset); // frees the match set list
}
