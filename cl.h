 /*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
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
#include "rule_dgp.h"
#include "rule_neural.h"

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
_Bool cl_crossover(CL *c1, CL *c2);
_Bool cl_general(CL *c1, CL *c2);
_Bool cl_match(CL *c, double *state);
_Bool cl_match_state(CL *c);
_Bool cl_mutate(CL *c);
_Bool cl_subsumer(CL *c);
_Bool cl_subsumes(CL *c1, CL *c2);
double cl_acc(CL *c);
double cl_del_vote(CL *c, double avg_fit);
double cl_predict(CL *c, double *state);
void cl_copy(CL *to, CL *from);
void cl_cover(CL *c, double *state);
void cl_free(CL *c);
void cl_init(CL *c, int size, int time);
void cl_print(CL *c);
void cl_rand(CL *c);
void cl_update(CL *c, double *state, double p, int set_num);
void cl_update_fit(CL *c, double acc_sum, double acc);

// classifier prediction
double pred_compute(CL *c, double *state);
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
#ifdef SAM
double cl_mutation_rate(CL *c, int m);
void sam_adapt(double *mu);       
void sam_copy(double *to, double *from);
void sam_free(double *mu);
void sam_init(double **mu);
void sam_print(double *mu);
#endif
