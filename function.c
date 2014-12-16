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
#include "random.h"
#include "cons.h"
#include "function.h"

double *state;

#define DATA_LENGTH 200
double data[DATA_LENGTH][8];
double answ[DATA_LENGTH];
int problem;

void func_init()
{
	// initialise problem function
#ifdef DATAFILE
	// read in (a small comma separated) data file
	// currently fixed to 8 input 1 output, 200 length
	// where the state values [0,20] and output [0,1]
	state_length = 8;
	char * infile = "in/star.dat";
	FILE *fin = fopen(infile, "rt");
	if(fin == 0) {
		printf("Error opening file: %s. %s.\n", infile, strerror(errno));
		exit(EXIT_FAILURE);
	} 
	char line[200];
	for(int i = 0; fgets(line, 110, fin) != NULL; i++) {
		data[i][0] = (atof(strtok(line, ",")) / 10.0) -1.0;
		for(int j = 1; j < 8; j++)
			data[i][j] = (atof(strtok(NULL, ",")) / 10.0) -1.0;
		answ[i] = (atof(strtok( NULL, ",")) * 2.0) -1.0;
	}
	fclose(fin);
	printf("Loaded input data file: %s\n", infile);
#else
	state_length = 1; // 1 input 1 output problem
	state = malloc(sizeof(double)*state_length);
#endif
}

double *func_state(_Bool train)
{
	// returns the problem input state
#ifdef DATAFILE
	if(train)
		problem = irand(0,180);
	else
		problem = irand(180,200);
	return data[problem]; 
#else
	// computed problem function
	for(int i = 0; i < state_length; i++)
		state[i] = (drand()*2.0) -1.0;
	return state;
#endif
}

double func_answer()
{
	// returns the problem solution
#ifdef DATAFILE
	return answ[problem];
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
	free(state);
}
