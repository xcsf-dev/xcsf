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
 * mechanisms to retrieve the next problem instance and solution values. Reads
 * in a variable length comma separated data file with variable number of
 * parameters (with the last parameter on a data line used as the target
 * output. All input and output parameters in the data file must be normalised
 * in the range [-1,1].
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

#define NUM_TEST 20 // number of data entries for testing (at end of file)
#define MAX_DATA 100000
#define MAX_LINE_LENGTH 200
#define DELIM ","

double *data; // data file variables read from a file
int cur_prob; // index of the current problem instance
int num_prob; // number of problem instances in the data file
int num_vars; // number of problem input + output variables

double *state; // current problem instance input variables

void func_init(char *infile)
{
	// initialise problem function
	// read in (a small comma separated) data file
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
	state = malloc(sizeof(double)*state_length);
	data = malloc(sizeof(double)*num_vars*num_prob);
	for(int i = 0; fgets(line,MAX_LINE_LENGTH,fin) != NULL; i++) {
		data[i*num_vars] = atof(strtok(line, DELIM));
		for(int j = 1; j < num_vars; j++)
			data[i*num_vars+j] = atof(strtok(NULL, DELIM));
	}
	// close
	fclose(fin);
	printf("Loaded input data file: %s\n", infile);
	printf("%d data entries with %d input variables per entry\n", 
			num_prob, state_length);
}

double *func_state(_Bool train)
{
	// returns the problem input state
	if(train)
		cur_prob = irand(0,num_prob-NUM_TEST);
	else
		cur_prob = irand(num_prob-NUM_TEST,num_prob);
	for(int i = 0; i < state_length; i++)
		state[i] = data[cur_prob*num_vars+i];
	return state;
}

double func_answer()
{
	// returns the problem solution
	return data[cur_prob*num_vars+state_length];
}

void func_free()
{
	free(data);
	free(state);
}
