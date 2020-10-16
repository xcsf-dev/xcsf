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
 * @brief An implementation of dynamical GP graphs with fuzzy activations.
 */

#pragma once

#include "xcsf.h"

/**
 * @brief Parameters for initialising DGP graphs.
 */
struct ArgsDGP {
    int max_k; //!< Maximum number of connections a node may have
    int max_t; //!< Maximum number of update cycles
    int n; //!< Number of nodes in the graph
    int n_inputs; //!< Number of inputs to the graph
};

/**
 * @brief Dynamical GP graph data structure.
 */
struct Graph {
    int *connectivity; //!< Connectivity map
    double *state; //!< Current state of each node
    double *initial_state; //!< Initial node states
    double *tmp_state; //!< Temporary storage for synchronous update
    double *tmp_input; //!< Temporary storage for updating the graph
    int *function; //!< Node activation functions
    int n; //!< Number of nodes
    int t; //!< Number of cycles to run
    int klen; //!< Length of connectivity map
    int max_k; //!< Maximum number of connections a node may have
    int max_t; //!< Maximum number of update cycles
    int n_inputs; //!< Number of inputs to the graph
    double *mu; //!< Mutation rates
};

bool
graph_mutate(struct Graph *dgp);

double
graph_output(const struct Graph *dgp, const int IDX);

size_t
graph_load(struct Graph *dgp, FILE *fp);

size_t
graph_save(const struct Graph *dgp, FILE *fp);

void
graph_copy(struct Graph *dest, const struct Graph *src);

void
graph_free(const struct Graph *dgp);

void
graph_init(struct Graph *dgp, const struct ArgsDGP *args);

void
graph_print(const struct Graph *dgp);

void
graph_rand(struct Graph *dgp);

void
graph_reset(const struct Graph *dgp);

void
graph_update(const struct Graph *dgp, const double *inputs, const bool reset);

void
graph_args_init(struct ArgsDGP *args);

void
graph_args_print(const struct ArgsDGP *args);

size_t
graph_args_save(const struct ArgsDGP *args, FILE *fp);

size_t
graph_args_load(struct ArgsDGP *args, FILE *fp);

/* parameter setters */

static inline void
graph_param_set_max_k(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP MAX_K too small\n");
        args->max_k = 1;
    } else {
        args->max_k = a;
    }
}

static inline void
graph_param_set_max_t(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP MAX_T too small\n");
        args->max_t = 1;
    } else {
        args->max_t = a;
    }
}

static inline void
graph_param_set_n(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP N too small\n");
        args->n = 1;
    } else {
        args->n = a;
    }
}

static inline void
graph_param_set_n_inputs(struct ArgsDGP *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set DGP N_INPUTS too small\n");
        args->n_inputs = 1;
    } else {
        args->n_inputs = a;
    }
}
