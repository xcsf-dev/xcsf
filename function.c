/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
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
 * The problem function module.
 *
 * Initialises the problem function that XCSF is to learn, and provides
 * mechanisms to retrieve the next problem instance and solution values. Reads
 * in a variable length comma separated data file with variable number of
 * parameters (with the last parameter on a data line used as the target
 * output. All input and output parameters in the data file must be normalised
 * in the range [-1,1]. Train and test set data files must be named as 
 * follows: {name}_train.dat and {name}_test.dat. To run XCSF on the data the
 * name must be specified at run time: xcsf {name}.
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

#define MAX_DATA 100000
#define MAX_LINE_LENGTH 200
#define DELIM ","

double *train_data; // data file variables read from a file
double *test_data; // data file variables read from a file
int cur_prob; // index of the current problem instance
int num_train_prob; // number of training problem instances in the data file
int num_test_prob; // number of testing problem instances in the data file
int num_test; // number of data entries for testing (at end of file)
int num_vars; // number of problem input + output variables

double *state; // current problem instance input variables

void func_read(char *fname, double **data, int *num_prob, int *num_vars);

void func_init(char *infile)
{
	// read in the training data
	char name[200];
	sprintf(name, "in/%s_train.dat", infile);
	func_read(name, &train_data, &num_train_prob, &num_vars);
	// initialise state array
	state_length = num_vars-1; // last var is output
	state = malloc(sizeof(double)*state_length);
	// read in the testing data
	sprintf(name, "in/%s_test.dat", infile);
	func_read(name, &test_data, &num_test_prob, &num_vars);
}
 
void func_read(char *fname, double **data, int *num_prob, int *num_vars)
{
 	FILE *fin = fopen(fname, "rt");
	if(fin == 0) {
		printf("Error opening file: %s. %s.\n", fname, strerror(errno));
		exit(EXIT_FAILURE);
	}    
	// ascertain the file length and number of vars per line
	*num_prob = 0;
	*num_vars = 0;
	char line[MAX_LINE_LENGTH];
	while(fgets(line, MAX_LINE_LENGTH, fin) != NULL) {
		if(*num_prob > MAX_DATA) {
			printf("data file %s is too big; maximum: %d\n", fname, MAX_DATA);
			exit(EXIT_FAILURE);
		}        
		// use the first line to count the number of variables on a line
		if(*num_prob == 0) {
			char *ptok = strtok(line, DELIM);
			while(ptok != NULL) {
				if(strlen(ptok) > 0)
					(*num_vars)++;
				ptok = strtok(NULL, DELIM);
			}
		}
		// count number of lines
		(*num_prob)++;
	}
	// read data file to memory
	rewind(fin);
	*data = malloc(sizeof(double) * (*num_vars) * (*num_prob));
	for(int i = 0; fgets(line,MAX_LINE_LENGTH,fin) != NULL; i++) {
		(*data)[i * (*num_vars)] = atof(strtok(line, DELIM));
		for(int j = 1; j < *num_vars; j++)
			(*data)[i * (*num_vars)+j] = atof(strtok(NULL, DELIM));
	}
	fclose(fin);
	printf("Loaded: %s\n", fname);
	printf("%d data entries with %d input variables per entry\n",
			*num_prob, *num_vars-1);

}

double *func_state(_Bool train)
{
	// returns the problem input state
	if(train) {
		cur_prob = irand(0,num_train_prob);
		for(int i = 0; i < state_length; i++)
			state[i] = train_data[cur_prob*num_vars+i];
	}
	else {
		cur_prob = irand(0,num_test_prob);
		for(int i = 0; i < state_length; i++)
			state[i] = test_data[cur_prob*num_vars+i];
	}
	return state;
}

double func_answer(_Bool train)
{
	// returns the problem solution
	if(train)
		return train_data[cur_prob*num_vars+state_length];
	else
		return test_data[cur_prob*num_vars+state_length];
}

void func_free()
{
	free(train_data);
	free(test_data);
	free(state);
}
