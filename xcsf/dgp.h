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
 * @date 2016--2020.
 * @brief An implementation of dynamical GP graphs with fuzzy activation functions.
 */ 

#pragma once
                  
/**
 * @brief Dynamical GP graph data structure.
 */ 
typedef struct GRAPH {
    int *connectivity; //!< Connectivity map
    double *state; //!< Current internal state
    double *tmp; //!< Temporary storage for synchronous update
    double *initial_state; //!< Initial states
    int *function; //!< Node activation functions
    int n; //!< Number of nodes
    int t; //!< Number of cycles to run
    int klen; //!< Length of connectivity map
} GRAPH;

_Bool graph_crossover(const XCSF *xcsf, GRAPH *dgp1, GRAPH *dgp2);
_Bool graph_mutate(const XCSF *xcsf, GRAPH *dgp);
double graph_output(const XCSF *xcsf, const GRAPH *dgp, int i);
size_t graph_load(const XCSF *xcsf, GRAPH *dgp, FILE *fp);
size_t graph_save(const XCSF *xcsf, const GRAPH *dgp, FILE *fp);
void graph_copy(const XCSF *xcsf, GRAPH *to, const GRAPH *from);
void graph_free(const XCSF *xcsf, const GRAPH *dgp);
void graph_init(const XCSF *xcsf, GRAPH *dgp, int n);
void graph_print(const XCSF *xcsf, const GRAPH *dgp);
void graph_rand(const XCSF *xcsf, GRAPH *dgp);
void graph_reset(const XCSF *xcsf, const GRAPH *dgp);
void graph_update(const XCSF *xcsf, const GRAPH *dgp, const double *inputs);
