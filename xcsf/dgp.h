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
 * @brief An implementation of dynamical GP graphs with fuzzy activation
 * functions.
 */

#pragma once

#include "xcsf.h"

#define DGP_N_MU (3) //!< Number of DGP mutation rates

/**
 * @brief Dynamical GP graph data structure.
 */
typedef struct GRAPH {
    int *connectivity; //!< Connectivity map
    double *state; //!< Current state of each node
    double *initial_state; //!< Initial node states
    double *tmp_state; //!< Temporary storage for synchronous update
    double *tmp_input; // !< Temporary storage for updating the graph
    int *function; //!< Node activation functions
    int n; //!< Number of nodes
    int t; //!< Number of cycles to run
    int klen; //!< Length of connectivity map
    double mu[DGP_N_MU]; //!< Mutation rates
} GRAPH;

_Bool
graph_crossover(const struct XCSF *xcsf, struct GRAPH *dgp1,
                struct GRAPH *dgp2);

_Bool
graph_mutate(const struct XCSF *xcsf, struct GRAPH *dgp);

double
graph_output(const struct XCSF *xcsf, const struct GRAPH *dgp, int i);

size_t
graph_load(const struct XCSF *xcsf, struct GRAPH *dgp, FILE *fp);

size_t
graph_save(const struct XCSF *xcsf, const struct GRAPH *dgp, FILE *fp);

void
graph_copy(const struct XCSF *xcsf, struct GRAPH *dest,
           const struct GRAPH *src);

void
graph_free(const struct XCSF *xcsf, const struct GRAPH *dgp);

void
graph_init(const struct XCSF *xcsf, struct GRAPH *dgp, int n);

void
graph_print(const struct XCSF *xcsf, const struct GRAPH *dgp);

void
graph_rand(const struct XCSF *xcsf, struct GRAPH *dgp);

void
graph_reset(const struct XCSF *xcsf, const struct GRAPH *dgp);

void
graph_update(const struct XCSF *xcsf, const struct GRAPH *dgp,
             const double *inputs);
