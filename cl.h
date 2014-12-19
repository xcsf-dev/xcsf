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
typedef struct CL
{
	double *cond;
	int cond_length;
	int weights_length;
	double *weights;
#ifdef RLS_PREDICTION
	double *matrix;
#endif
	double pre;
	double err;
	double fit;
	int num;
	int exp;
	double size;
	int time;
#ifdef SELF_ADAPT_MUTATION
	double *mu;
	int *iset;
	double *gset;
#endif
} CL;

// general classifier
_Bool cl_subsumer(CL *c);
double cl_acc(CL *c);
double cl_del_vote(CL *c, double avg_fit);
double cl_update_size(CL *c, double num_sum);
void cl_copy(CL *to, CL *from);
void cl_free(CL *c);
void cl_init(CL *c, int size, int time);
void cl_print(CL *c);
void cl_update_fit(CL *c, double acc_sum, double acc);

// classifier prediction
double pred_compute(CL *c, double *state);
double pred_update_err(CL *c, double p);
void pred_copy(CL *to, CL *from);
void pred_free(CL *c);
void pred_init(CL *c);
void pred_print(CL *c);
void pred_update(CL *c, double p, double *state);

// classifier condition
_Bool cond_crossover(CL *c1, CL *c2);
_Bool cond_general(CL *c1, CL *c2);
_Bool cond_match(CL *c, double *state);
_Bool cond_mutate(CL *c);
_Bool cond_subsumes(CL *c1, CL *c2);
void cond_copy(CL *to, CL *from);
void cond_cover(CL *c, double *state);
void cond_free(CL *c);
void cond_init(CL *c);
void cond_print(CL *c);
void cond_rand(CL *c);

// self-adaptive mutation
void sam_adapt(CL *c);       
void sam_copy(CL *to, CL *from);
void sam_free(CL *c);
void sam_init(CL *c);
