/*
 * Copyright (C) 2016--2019 Richard Preen <rpreen@gmail.com>
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

#define GP_MAX_LEN 10000
#define GP_DEPTH 5
#define GP_NUM_FUNC 4
#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3
  
typedef struct GP_TREE {
	int *tree;
	int p;
} GP_TREE;
 
void tree_free_cons(XCSF *xcsf);
void tree_init_cons(XCSF *xcsf);
void tree_init(XCSF *xcsf, GP_TREE *gp);
void tree_free(XCSF *xcsf, GP_TREE *gp);
void tree_rand(XCSF *xcsf, GP_TREE *gp);
void tree_copy(XCSF *xcsf, GP_TREE *to, GP_TREE *from);
int tree_print(XCSF *xcsf, GP_TREE *gp, int p);
double tree_eval(XCSF *xcsf, GP_TREE *gp, double *x);
void tree_crossover(XCSF *xcsf, GP_TREE *p1, GP_TREE *p2);
void tree_mutation(XCSF *xcsf, GP_TREE *offspring, double rate);
