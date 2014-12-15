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
#include <time.h>
#include "cons.h"
#include "random.h"
#include "cl.h"
#include "cl_set.h"
#include "bpn.h"
#include "function.h"
#include "gplot.h"

void disp_perf(double *error, int trial);

FILE *fout;
char fname[30];

int main(int argc, char *argv[0])
{    
	if(argc < 1 || argc > 3) {
		printf("Usage: xcsf [MaxTrials] [NumExp]\n");
		exit(EXIT_FAILURE);
	} 
	// file for writing output; uses the date/time/exp as file name
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	char basefname[30];
	sprintf(basefname, "out/%04d-%02d-%02d-%02d%02d%02d", tm.tm_year + 1900, 
			tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

	// initilise environment
	init_constants(argc, argv);
	init_random();
	func_init();
#ifdef NEURAL_CONDITIONS
	// classifiers currently fixed to 3 layer networks
	int neurons[3] = {state_length, NUM_HIDDEN_NEURONS, 1};
	neural_init(3, neurons);
#endif

	// run experiments
	double err[PERF_AVG_TRIALS];
	for(int e = 1; e < NUM_EXPERIMENTS+1; e++) {
		// create output file
		sprintf(fname, "%s-%d.dat", basefname, e);
		fout = fopen(fname, "wt");
		if(fout == 0) {
			printf("Error opening file: %s. %s.\n", fname, strerror(errno));
			exit(EXIT_FAILURE);
		} 
#ifdef GNUPLOT
		gplot_init(fname);
#endif
		printf("\nExperiment: %d\n", e);

		init_pop();
		// each trial in the experiment
		for(int trial = 0; trial < MAX_TRIALS; trial++) {
			// get problem function state and solution
			double *state = func_state();
			double answer = func_answer();
			// create match set
			NODE *mset = NULL, *kset = NULL;
			int msize = 0, mnum = 0;
			match_set(&mset, &msize, &mnum, state, trial, &kset);
			// calculate system prediction and track performance
			double pre = weighted_pred(&mset, state);
			double abserr = fabs(answer - pre);
			err[trial%PERF_AVG_TRIALS] = abserr;
			if(trial%PERF_AVG_TRIALS == 0 && trial > 0)
				disp_perf(err, trial);
			// provide reinforcement to the set
			update_set(&mset, &msize, &mnum, answer, &kset, state);
			// run the genetic algorithm
			ga(&mset, msize, mnum, trial, &kset);
			// clean up
			clean_set(&kset, &mset, true);
			free_set(&mset);       
		}
		kill_set(&pset);
		fclose(fout);
#ifdef GNUPLOT
		gplot_close();
#endif
	}
	func_free();
#ifdef NEURAL_CONDITIONS
	neural_free();
#endif
	return EXIT_SUCCESS;
}

void disp_perf(double *error, int trial)
{
	double serr = 0.0;
	for(int i = 0; i < PERF_AVG_TRIALS; i++)
		serr += error[i];
	serr /= (double)PERF_AVG_TRIALS;
	printf("%d %.5f %d", trial, serr, pop_num);
	fprintf(fout, "%d %.5f %d", trial, serr, pop_num);
#ifdef SELF_ADAPT_MUTATION
	for(int i = 0; i < NUM_MU; i++) {
		printf(" %.5f", avg_mut(&pset, i));
		fprintf(fout, " %.5f", avg_mut(&pset, i));
	}
#endif
	printf("\n");
	fprintf(fout, "\n");
	fflush(stdout);
	fflush(fout);
#ifdef GNUPLOT
	gplot_draw();
#endif
}
