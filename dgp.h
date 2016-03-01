/*
 * Copyright (C) 2016 Richard Preen <rpreen@gmail.com>
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
  
#define MAX_T 10 // maximum number of cycles to update graph
#define NUM_FUNC 7 // number of node available functions
#define MAX_K 2 // maximum inputs to a node
 
typedef struct GNODE {
	int conn[MAX_K]; // connectivity map to other nodes
	int k; // number of inputs
	double state; // current internal state
	double initial_state; // initial state
	int func;  // arithmetic function
} GNODE;

typedef struct GRAPH {
	double real_error;
	double fitness;
	int n; // number of nodes in this graph
	int t; // number of cycles to run
	GNODE *nodes; // nodes
} GRAPH;

void graph_init(GRAPH *dgp, int n);
void graph_free(GRAPH *dgp);
void graph_rand(GRAPH *dgp);
void graph_print(GRAPH *dgp);
void graph_copy(GRAPH *to, GRAPH *from);
_Bool graph_mutate(GRAPH *dgp, double rate);
void graph_update(GRAPH *dgp, double *inputs);
double graph_output(GRAPH *dgp, int i);
void graph_reset(GRAPH *dgp);
double graph_avg_k(GRAPH *dgp);
