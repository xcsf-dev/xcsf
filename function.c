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

void func_init()
{
	// initialise problem function
	state_length = 1; // 1 input 1 output problem
	state = malloc(sizeof(double)*state_length);
}

double *func_state()
{
	// returns the problem input state
	for(int i = 0; i < state_length; i++)
		state[i] = (drand()*2.0) -1.0;
	return state;
}

double func_answer()
{
	// returns the problem solution
	// sine function problem
	double answer = 0.0;
	for(int i = 0; i < state_length; i++) {
		state[i] = (drand()*2.0) -1.0;
		answer += state[i];
	}
	answer *= 4.0 * M_PI;
	answer = sin(answer);


//	// sextic polynomial function problem
//	for(int i = 0; i < state_length; i++)
//		state[i] = (drand()*2.0) -1.0;
//	double answer = pow(state[0],6)+(2*pow(state[0],4))+pow(state[0],2);

	return answer;
}

void func_free()
{
	free(state);
}
