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
 
NODE *pop_del();
double set_mean_time(NODE **set, int num_sum);
double set_pred(NODE **set, int size, double *state);
double set_total_fit(NODE **set);
double set_total_time(NODE **set);
void pop_add(CL *c);
void pop_init();
void pop_enforce_limit(NODE **kset);
void set_add(NODE **set, CL *c);
void set_free(NODE **set);
void set_kill(NODE **set);
void set_match(NODE **set, int *size, int *num, double *state, int time, NODE **kset);
void set_print(NODE *set);
void set_times(NODE **set, int time);
void set_update(NODE **set, int *size, int *num, double r, NODE **kset, double *state);
void set_validate(NODE **set, int *size, int *num);
#ifdef SELF_ADAPT_MUTATION
double set_avg_mut(NODE **set, int m);
#endif

NODE *pset;
int pop_num;
int pop_num_sum;
