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
 * The problem function module.
 *
 * Initialises the problem function that XCSF is to learn, and provides
 * mechanisms to retrieve the next problem instance and solution values. Reads
 * in a variable length comma separated data file with variable number of
 * parameters (with the last parameter on a data line used as the target
 * output. All input and output parameters in the data file must be normalised
 * in the range [-1,1]. 
 * Train and test set data files must be named as follows:
 * {name}_{train|test}_{x|y}.csv with input variables x and labelled outputs y. 
 * To run XCSF on the data the name must be specified at run time: xcsf {name}.
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

double *train_x; // data file variables read from file
double *train_y;
double *test_x;
double *test_y;
int num_train_prob; // number of training problem instances in the data file
int num_test_prob; // number of testing problem instances in the data file
int num_test; // number of data entries for testing (at end of file)

void func_read(char *fname, double **data, int *num_prob, int *num_vars);

void func_init(char *infile)
{
	char name[200];
	// read the input variables
	sprintf(name, "in/%s_train_x.csv", infile);
	func_read(name, &train_x, &num_train_prob, &num_x_vars);
	sprintf(name, "in/%s_test_x.csv", infile);
	func_read(name, &test_x, &num_test_prob, &num_x_vars);
	// read the output variables
 	sprintf(name, "in/%s_train_y.csv", infile);
	func_read(name, &train_y, &num_train_prob, &num_y_vars);
	sprintf(name, "in/%s_test_y.csv", infile);
	func_read(name, &test_y, &num_test_prob, &num_y_vars);
}
 
void func_read(char *fname, double **data, int *num_rows, int *num_cols)
{
	// Provided a file name: will set the data, num_rows, num_cols 
 	FILE *fin = fopen(fname, "rt");
	if(fin == 0) {
		printf("Error opening file: %s. %s.\n", fname, strerror(errno));
		exit(EXIT_FAILURE);
	}    
	// ascertain the file length and number of variables per line
	*num_rows = 0;
	*num_cols = 0;
	char line[MAX_LINE_LENGTH];
	while(fgets(line, MAX_LINE_LENGTH, fin) != NULL) {
		if(*num_rows > MAX_DATA) {
			printf("data file %s is too big; maximum: %d\n", fname, MAX_DATA);
			exit(EXIT_FAILURE);
		}        
		// use the first line to count the number of variables on a line
		if(*num_rows == 0) {
			char *ptok = strtok(line, DELIM);
			while(ptok != NULL) {
				if(strlen(ptok) > 0)
					(*num_cols)++;
				ptok = strtok(NULL, DELIM);
			}
		}
		// count number of lines
		(*num_rows)++;
	}
	// read data file to memory
	rewind(fin);
	*data = malloc(sizeof(double) * (*num_cols) * (*num_rows));
	for(int i = 0; fgets(line,MAX_LINE_LENGTH,fin) != NULL; i++) {
		(*data)[i * (*num_cols)] = atof(strtok(line, DELIM));
		for(int j = 1; j < *num_cols; j++)
			(*data)[i * (*num_cols)+j] = atof(strtok(NULL, DELIM));
	}
	fclose(fin);
	printf("Loaded: %s: %d rows, %d cols\n", fname, *num_rows, *num_cols);
}

void func_rand_sample(double *x, double *y, _Bool train)
{
	if(train) {
		int cur_prob = irand(0,num_train_prob);
		for(int i = 0; i < num_x_vars; i++) {
			x[i] = train_x[cur_prob*num_x_vars+i];
		}
		for(int i = 0; i < num_y_vars; i++) {
			y[i] = train_y[cur_prob*num_y_vars+i];
		}
	}
	else {
		int cur_prob = irand(0,num_test_prob);
		for(int i = 0; i < num_x_vars; i++) {
			x[i] = test_x[cur_prob*num_x_vars+i];
		}
		for(int i = 0; i < num_y_vars; i++) {
			y[i] = test_y[cur_prob*num_y_vars+i];
		}
	}
}

void func_free()
{
	free(train_x);
	free(train_y);
	free(test_x);
	free(test_y);
}
