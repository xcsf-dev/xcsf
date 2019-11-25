/*
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
 */
  
/**
 * @file dgp.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2019.
 * @brief An implementation of dynamical GP graphs
 */ 

#pragma once
                  
/**
 * @brief Dynamical GP graph data structure.
 */ 
typedef struct GRAPH {
    int *connectivity; //!< Connectivity map
    double *weights; //!< Connection weights
    double *state; //!< Current internal state
    double *initial_state; //!< Initial states
    int *function; //!< Node activation functions
    int n; //!< Number of nodes
    int t; //!< Number of cycles to run
    int klen; //!< Length of connectivity map
} GRAPH;

_Bool graph_crossover(XCSF *xcsf, GRAPH *dgp1, GRAPH *dgp2);
_Bool graph_mutate(XCSF *xcsf, GRAPH *dgp);
double graph_output(XCSF *xcsf, GRAPH *dgp, int i);
size_t graph_load(XCSF *xcsf, GRAPH *dgp, FILE *fp);
size_t graph_save(XCSF *xcsf, GRAPH *dgp, FILE *fp);
void graph_copy(XCSF *xcsf, GRAPH *to, GRAPH *from);
void graph_free(XCSF *xcsf, GRAPH *dgp);
void graph_init(XCSF *xcsf, GRAPH *dgp, int n);
void graph_print(XCSF *xcsf, GRAPH *dgp);
void graph_rand(XCSF *xcsf, GRAPH *dgp);
void graph_reset(XCSF *xcsf, GRAPH *dgp);
void graph_update(XCSF *xcsf, GRAPH *dgp, double *inputs);
