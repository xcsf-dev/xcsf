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
 * @file gp.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2020.
 * @brief An implementation of GP trees based upon TinyGP.
 */

#pragma once

#include "xcsf.h"

/**
 * @brief Parameters for initialising GP trees.
 */
struct ArgsGPTree {
    double max; //!< Maximum value of a constant
    double min; //!< Minimum value of a constant
    int n_inputs; //!< Number of inputs
    int n_constants; //!< Number of constants available
    int init_depth; //!< Initial depth
    int max_len; //!< Maximum initial length
    double *constants; //!< Constants available for GP trees
};

/**
 * @brief GP tree data structure.
 */
struct GPTree {
    int *tree; //!< Flattened tree representation of functions and terminals
    int len; //!< Size of the tree
    int p; //!< Current position in the tree
    double *mu; //!< Mutation rates
};

void
tree_free(const struct GPTree *gp);

void
tree_rand(struct GPTree *gp, const struct ArgsGPTree *args);

void
tree_copy(struct GPTree *dest, const struct GPTree *src);

int
tree_print(const struct GPTree *gp, const struct ArgsGPTree *args, int p);

double
tree_eval(struct GPTree *gp, const struct ArgsGPTree *args, const double *x);

void
tree_crossover(struct GPTree *p1, struct GPTree *p2);

bool
tree_mutate(struct GPTree *gp, const struct ArgsGPTree *args);

size_t
tree_save(const struct GPTree *gp, FILE *fp);

size_t
tree_load(struct GPTree *gp, FILE *fp);

void
tree_args_init(struct ArgsGPTree *args);

void
tree_args_free(struct ArgsGPTree *args);

void
tree_args_print(const struct ArgsGPTree *args);

size_t
tree_args_save(const struct ArgsGPTree *args, FILE *fp);

size_t
tree_args_load(struct ArgsGPTree *args, FILE *fp);

void
tree_args_init_constants(struct ArgsGPTree *args);

/* parameter setters */

static inline void
tree_param_set_max(struct ArgsGPTree *args, const double a)
{
    args->max = a;
}

static inline void
tree_param_set_min(struct ArgsGPTree *args, const double a)
{
    args->min = a;
}

static inline void
tree_param_set_n_inputs(struct ArgsGPTree *args, const int a)
{
    args->n_inputs = a;
}

static inline void
tree_param_set_n_constants(struct ArgsGPTree *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP N_CONSTANTS too small\n");
        args->n_constants = 1;
    } else {
        args->n_constants = a;
    }
}

static inline void
tree_param_set_init_depth(struct ArgsGPTree *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP INIT_DEPTH too small\n");
        args->init_depth = 1;
    } else {
        args->init_depth = a;
    }
}

static inline void
tree_param_set_max_len(struct ArgsGPTree *args, const int a)
{
    if (a < 1) {
        printf("Warning: tried to set GP MAX_LEN too small\n");
        args->max_len = 1;
    } else {
        args->max_len = a;
    }
}
