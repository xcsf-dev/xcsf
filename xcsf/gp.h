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
 * @brief GP tree data structure.
 */
struct GPTree {
    int *tree; //!< Flattened tree representation of functions and terminals
    int len; //!< Size of the tree
    int p; //!< Current position in the tree
    double *mu; //!< Mutation rates
};

void
tree_free_cons(const struct XCSF *xcsf);

void
tree_init_cons(struct XCSF *xcsf);

void
tree_free(const struct XCSF *xcsf, const struct GPTree *gp);

void
tree_rand(const struct XCSF *xcsf, struct GPTree *gp);

void
tree_copy(const struct XCSF *xcsf, struct GPTree *dest,
          const struct GPTree *src);

int
tree_print(const struct XCSF *xcsf, const struct GPTree *gp, int p);

double
tree_eval(const struct XCSF *xcsf, struct GPTree *gp, const double *x);

void
tree_crossover(const struct XCSF *xcsf, struct GPTree *p1, struct GPTree *p2);

bool
tree_mutate(const struct XCSF *xcsf, struct GPTree *gp);

size_t
tree_save(const struct XCSF *xcsf, const struct GPTree *gp, FILE *fp);

size_t
tree_load(const struct XCSF *xcsf, struct GPTree *gp, FILE *fp);
