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

#define GP_N_MU (1) //!< Number of GP mutation rates

/**
 * @brief GP tree data structure.
 */
typedef struct GP_TREE {
    int *tree; //!< Flattened tree representation of functions and terminals
    int len; //!< Size of the tree
    int p; //!< Current position in the tree
    double mu[GP_N_MU]; //!< Mutation rates
} GP_TREE;

void tree_free_cons(const XCSF *xcsf);
void tree_init_cons(XCSF *xcsf);
void tree_free(const XCSF *xcsf, const GP_TREE *gp);
void tree_rand(const XCSF *xcsf, GP_TREE *gp);
void tree_copy(const XCSF *xcsf, GP_TREE *dest, const GP_TREE *src);
int tree_print(const XCSF *xcsf, const GP_TREE *gp, int p);
double tree_eval(const XCSF *xcsf, GP_TREE *gp, const double *x);
void tree_crossover(const XCSF *xcsf, GP_TREE *p1, GP_TREE *p2);
_Bool tree_mutate(const XCSF *xcsf, GP_TREE *gp);
size_t tree_save(const XCSF *xcsf, const GP_TREE *gp, FILE *fp);
size_t tree_load(const XCSF *xcsf, GP_TREE *gp, FILE *fp);
