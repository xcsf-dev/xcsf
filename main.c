/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include "cons.h"
#include "random.h"
#include "cl.h"
#include "cl_set.h"
#include "neural.h"
#include "function.h"
#include "perf.h"

void trial(int trial, _Bool train, double *err);

int main(int argc, char *argv[0])
{    
	if(argc < 1 || argc > 3) {
		printf("Usage: xcsf [MaxTrials] [NumExp]\n");
		exit(EXIT_FAILURE);
	} 

	// initialise environment
	constants_init(argc, argv);
	random_init();
	func_init();
	gen_outfname();
#ifdef NEURAL_CONDITIONS
	// classifiers currently fixed to 3 layer networks
	int neurons[3] = {state_length, NUM_HIDDEN_NEURONS, 1};
	neural_init(3, neurons);
#endif

	// run experiments
	double err[PERF_AVG_TRIALS];
	double terr[PERF_AVG_TRIALS];
	for(int e = 1; e < NUM_EXPERIMENTS+1; e++) {
		printf("\nExperiment: %d\n", e);
		pop_init();
		outfile_init(e);
		// each trial in the experiment
		for(int cnt = 0; cnt < MAX_TRIALS; cnt++) {
			trial(cnt, true, err); // train
			trial(cnt, false, terr);// test
			// display performance
			if(cnt%PERF_AVG_TRIALS == 0 && cnt > 0)
				disp_perf(err, terr, cnt, pop_num);
		}
		set_kill(&pset);
		outfile_close();
	}
	func_free();
#ifdef NEURAL_CONDITIONS
	neural_free();
#endif
	return EXIT_SUCCESS;
}

void trial(int cnt, _Bool train, double *err)
{
	// get problem function state and solution
	double *state = func_state(train);
	double answer = func_answer();
	// create match set
	NODE *mset = NULL, *kset = NULL;
	int msize = 0, mnum = 0;
	set_match(&mset, &msize, &mnum, state, cnt, &kset);
	// calculate system prediction and track performance
	double pre = set_pred(&mset, state);
	double abserr = fabs(answer - pre);
	err[cnt%PERF_AVG_TRIALS] = abserr;
	if(train) {
		// provide reinforcement to the set
		set_update(&mset, &msize, &mnum, answer, &kset, state);
		// run the genetic algorithm
		ga(&mset, msize, mnum, cnt, &kset);
	}
	// clean up
	set_clean(&kset, &mset, true);
	set_free(&mset);    
}
