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
 *
 **************
 * Description: 
 **************
 * The problem function module.
 *
 * Initialises the problem function that XCSF is to learn, and provides
 * mechanisms to retrieve the next problem instance and solution values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include "random.h"
#include "cons.h"
#include "function.h"

#ifdef DATAFILE
#define NUM_TEST 20 // number of data entries for testing (at end of file)
#define MAX_DATA 100000
#define MAX_LINE_LENGTH 200
#define DELIM ","
double *data; // data file variables read from a file
int cur_prob; // index of the current problem instance
int num_prob; // number of problem instances in the data file
int num_vars; // number of problem input + output variables
#endif

double *state; // current problem instance input variables

void func_init()
{
	// initialise problem function
#ifdef DATAFILE
	// read in (a small comma separated) data file
	char * infile = "in/star.dat";
	FILE *fin = fopen(infile, "rt");
	if(fin == 0) {
		printf("Error opening file: %s. %s.\n", infile, strerror(errno));
		exit(EXIT_FAILURE);
	}    
	// ascertain the file length and number of vars per line
	char line[MAX_LINE_LENGTH];
	for(num_prob = 0; fgets(line, MAX_LINE_LENGTH, fin) != NULL; num_prob++) {
		// number of lines
		if(num_prob > MAX_DATA) {
			printf("input data file is too big; maximum: %d\n", MAX_DATA);
			exit(EXIT_FAILURE);
		}        
		// use the first line to count the number of variables on a line
		if(num_prob == 0) {
			char *ptok = strtok(line, DELIM);
			while(ptok != NULL) {
				if(strlen(ptok) > 0)
					num_vars++;
				ptok = strtok(NULL, DELIM);
			}
		}
	}
	// read data file to memory
	rewind(fin);
	state_length = num_vars-1; // last var is output
	data = malloc(sizeof(double)*num_vars*num_prob);
	for(int i = 0; fgets(line,MAX_LINE_LENGTH,fin) != NULL; i++) {
		// read input vars
		data[i*num_vars] = (atof(strtok(line,DELIM)) /10.0)-1.0;
		for(int j = 1; j < state_length; j++)
			data[i*num_vars+j] = (atof(strtok(NULL, DELIM)) /10.0)-1.0;
		// read output var
		data[i*num_vars+state_length] = (atof(strtok(NULL, DELIM)) *2.0)-1.0;
	}
	// close
	fclose(fin);
	printf("Loaded input data file: %s\n", infile);
	printf("%d data entries with %d input variables per entry\n", 
			num_prob, state_length);
#else
	// for computed problems
	state_length = 1; // 1 input 1 output problem
#endif
	state = malloc(sizeof(double)*state_length);
}

double *func_state(_Bool train)
{
	// returns the problem input state
#ifdef DATAFILE
	if(train)
		cur_prob = irand(0,num_prob-NUM_TEST);
	else
		cur_prob = irand(num_prob-NUM_TEST,num_prob);
	for(int i = 0; i < state_length; i++)
		state[i] = data[cur_prob*num_vars+i];
#else
	// computed problem function
	for(int i = 0; i < state_length; i++)
		state[i] = (drand()*2.0) -1.0;
#endif
	return state;
}

double func_answer()
{
	// returns the problem solution
#ifdef DATAFILE
	return data[cur_prob*num_vars+state_length];
#else
	// computed sine function problem
	double answer = 0.0;
	for(int i = 0; i < state_length; i++) {
		state[i] = (drand()*2.0) -1.0;
		answer += state[i];
	}
	answer *= 4.0 * M_PI;
	answer = sin(answer);
	return answer;

	//	// computed sextic polynomial function problem
	//	for(int i = 0; i < state_length; i++)
	//		state[i] = (drand()*2.0) -1.0;
	//	double answer = pow(state[0],6)+(2*pow(state[0],4))+pow(state[0],2);
	//	return answer;
#endif
}

void func_free()
{
#ifdef DATAFILE
	free(data);
#endif
	free(state);
}
