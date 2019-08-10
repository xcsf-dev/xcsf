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

typedef struct GRAPH {
	int *connectivity; // connectivity map
	double *state; // current internal state
	double *initial_state; // initial states
	int *function; // node functions
	int n; // number of nodes
	int t; // number of cycles to run
	double avgk; // average number of active connections
} GRAPH;

void graph_init(XCSF *xcsf, GRAPH *dgp, int n);
void graph_free(XCSF *xcsf, GRAPH *dgp);
void graph_rand(XCSF *xcsf, GRAPH *dgp);
void graph_print(XCSF *xcsf, GRAPH *dgp);
void graph_copy(XCSF *xcsf, GRAPH *to, GRAPH *from);
_Bool graph_mutate(XCSF *xcsf, GRAPH *dgp, double rate);
void graph_update(XCSF *xcsf, GRAPH *dgp, double *inputs);
double graph_output(XCSF *xcsf, GRAPH *dgp, int i);
void graph_reset(XCSF *xcsf, GRAPH *dgp);
double graph_avg_k(XCSF *xcsf, GRAPH *dgp);
