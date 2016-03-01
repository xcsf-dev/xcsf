 /*
 * Copyright (C) 2015--2016 Richard Preen <rpreen@gmail.com>
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

#include "neural.h"
#include "gp.h"
#include "dgp.h"
#include "cond_rect.h"
#include "cond_neural.h"
#include "cond_gp.h"
#include "cond_dgp.h"
#include "pred_nlms.h"
#include "pred_rls.h"
#include "pred_neural.h"

typedef struct CL {
	COND cond;
	PRED pred;
	double err;
	double fit;
	int num;
	int exp;
	double size;
	int time;
} CL;

// general classifier
_Bool cl_subsumer(CL *c);
double cl_acc(CL *c);
double cl_del_vote(CL *c, double avg_fit);
void cl_copy(CL *to, CL *from);
void cl_free(CL *c);
void cl_init(CL *c, int size, int time);
void cl_print(CL *c);
void cl_update(CL *c, double *state, double p, int set_num);
void cl_update_fit(CL *c, double acc_sum, double acc);

// classifier prediction
double pred_compute(PRED *pred, double *state);
void pred_copy(PRED *to, PRED *from);
void pred_free(PRED *pred);
void pred_init(PRED *pred);
void pred_print(PRED *pred);
void pred_update(PRED *pred, double p, double *state);

// classifier condition
_Bool cond_crossover(COND *cond1, COND *cond2);
_Bool cond_general(COND *cond1, COND *cond2);
_Bool cond_match(COND *cond, double *state);
_Bool cond_mutate(COND *cond);
_Bool cond_subsumes(COND *cond1, COND *cond2);
void cond_copy(COND *to, COND *from);
void cond_cover(COND *cond, double *state);
void cond_free(COND *cond);
void cond_init(COND *cond);
void cond_print(COND *cond);
void cond_rand(COND *cond);

// self-adaptive mutation
void sam_adapt(double *mu);       
void sam_copy(double *to, double *from);
void sam_free(double *mu);
void sam_init(double **mu);
void sam_print(double *mu);
