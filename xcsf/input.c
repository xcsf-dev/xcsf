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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include "data_structures.h"
#include "random.h"
#include "input.h"

#define MAX_DATA 100000
#define MAX_LINE_LENGTH 200
#define DELIM ","

#define PATH_TO_DATA "../../data"

void csv_read(char *fname, double **data, int *num_prob, int *num_vars);

void input_read_csv(char *infile, INPUT_DATA *train_data, INPUT_DATA *test_data)
{
	// expects an identical number of x and y rows
	char name[200];
	sprintf(name, "%s/%s_train_x.csv", PATH_TO_DATA, infile);
	csv_read(name, &train_data->x, &train_data->rows, &train_data->x_cols);
	sprintf(name, "%s/%s_train_y.csv", PATH_TO_DATA, infile);
	csv_read(name, &train_data->y, &train_data->rows, &train_data->y_cols);
	sprintf(name, "%s/%s_test_x.csv", PATH_TO_DATA, infile);
	csv_read(name, &test_data->x, &test_data->rows, &test_data->x_cols);
	sprintf(name, "%s/%s_test_y.csv", PATH_TO_DATA, infile);
	csv_read(name, &test_data->y, &test_data->rows, &test_data->y_cols);
}
 
void csv_read(char *fname, double **data, int *num_rows, int *num_cols)
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
				if(strlen(ptok) > 0) {
					(*num_cols)++;
				}
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
		for(int j = 1; j < *num_cols; j++) {
			(*data)[i * (*num_cols)+j] = atof(strtok(NULL, DELIM));
		}
	}
	fclose(fin);
	printf("Loaded: %s: %d rows, %d cols\n", fname, *num_rows, *num_cols);
}

void input_rand_sample(INPUT_DATA *data, double *x, double *y)
{
	int idx = irand(0,data->rows);
	for(int i = 0; i < data->x_cols; i++) {
		x[i] = data->x[idx * data->x_cols + i];
	}
	for(int i = 0; i < data->y_cols; i++) {
		y[i] = data->y[idx * data->y_cols + i];
	}
}

void input_free(INPUT_DATA *data)
{                 
	free(data->x);
	free(data->y);
}
