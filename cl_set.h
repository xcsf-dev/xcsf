/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
typedef struct NODE
{
	CL *cl;
	struct NODE *next;
} NODE;
 
void init_pop();
double weighted_pred(NODE **set, double *state);
void match_set(NODE **set, int *size, int *num, double *state, int time, NODE **kset);
void ga(NODE **set, int size, int num, int time, NODE **kset);
void set_validate(NODE **set, int *size, int *num);
void print_set(NODE *set);
void free_set(NODE **set);
void kill_set(NODE **set);
void clean_set(NODE **kset, NODE **set, _Bool in_set);
void update_set(NODE **set, int *size, int *num, double r, NODE **kset, double *state);
#ifdef SELF_ADAPT_MUTATION
double avg_mut(NODE **set, int m);
#endif

NODE *pset;
int pop_num;
int pop_num_sum;
