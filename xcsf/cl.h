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


// classifier condition

struct CondVtbl {
	_Bool (*cond_impl_crossover)(XCSF *xcsf, CL *c1, CL *c2);
	_Bool (*cond_impl_general)(XCSF *xcsf, CL *c1, CL *c2);
	_Bool (*cond_impl_match)(XCSF *xcsf, CL *c, double *x);
	_Bool (*cond_impl_match_state)(XCSF *xcsf, CL *c);
	_Bool (*cond_impl_mutate)(XCSF *xcsf, CL *c);
	double (*cond_impl_mu)(XCSF *xcsf, CL *c, int m);
	void (*cond_impl_copy)(XCSF *xcsf, CL *to, CL *from);
	void (*cond_impl_cover)(XCSF *xcsf, CL *c, double *x);
	void (*cond_impl_free)(XCSF *xcsf, CL *c);
	void (*cond_impl_init)(XCSF *xcsf, CL *c);
	void (*cond_impl_print)(XCSF *xcsf, CL *c);
	void (*cond_impl_rand)(XCSF *xcsf, CL *c);
};

static inline _Bool cond_crossover(XCSF *xcsf, CL *c1, CL *c2) {
	return (*c1->cond_vptr->cond_impl_crossover)(xcsf, c1, c2);
}

static inline _Bool cond_general(XCSF *xcsf, CL *c1, CL *c2) {
	return (*c1->cond_vptr->cond_impl_general)(xcsf, c1, c2);
}

static inline _Bool cond_match(XCSF *xcsf, CL *c, double *x) {
	return (*c->cond_vptr->cond_impl_match)(xcsf, c, x);
}

static inline _Bool cond_match_state(XCSF *xcsf, CL *c) {
	return (*c->cond_vptr->cond_impl_match_state)(xcsf, c);
}

static inline _Bool cond_mutate(XCSF *xcsf, CL *c) {
	return (*c->cond_vptr->cond_impl_mutate)(xcsf, c);
}

static inline double cond_mu(XCSF *xcsf, CL *c, int m) {
	return (*c->cond_vptr->cond_impl_mu)(xcsf, c, m);
}

static inline void cond_copy(XCSF *xcsf, CL *to, CL *from) {
	(*to->cond_vptr->cond_impl_copy)(xcsf, to, from);
}

static inline void cond_cover(XCSF *xcsf, CL *c, double *x) {
	(*c->cond_vptr->cond_impl_cover)(xcsf, c, x);
}

static inline void cond_free(XCSF *xcsf, CL *c) {
	(*c->cond_vptr->cond_impl_free)(xcsf, c);
}

static inline void cond_init(XCSF *xcsf, CL *c) {
	(*c->cond_vptr->cond_impl_init)(xcsf, c);
}

static inline void cond_print(XCSF *xcsf, CL *c) {
	(*c->cond_vptr->cond_impl_print)(xcsf, c);
}

static inline void cond_rand(XCSF *xcsf, CL *c) {
	(*c->cond_vptr->cond_impl_rand)(xcsf, c);
}

// classifier prediction    

struct PredVtbl {
	double *(*pred_impl_compute)(XCSF *xcsf, CL *c, double *x);
	double (*pred_impl_pre)(XCSF *xcsf, CL *c, int p);
	void (*pred_impl_copy)(XCSF *xcsf, CL *to,  CL *from);
	void (*pred_impl_free)(XCSF *xcsf, CL *c);
	void (*pred_impl_init)(XCSF *xcsf, CL *c);
	void (*pred_impl_print)(XCSF *xcsf, CL *c);
	void (*pred_impl_update)(XCSF *xcsf, CL *c, double *x, double *y);
};

static inline double *pred_compute(XCSF *xcsf, CL *c, double *x) {
	return (*c->pred_vptr->pred_impl_compute)(xcsf, c, x);
}

static inline double pred_pre(XCSF *xcsf, CL *c, int p) {
	return (*c->pred_vptr->pred_impl_pre)(xcsf, c, p);
}

static inline void pred_copy(XCSF *xcsf, CL *to, CL *from) {
	(*to->pred_vptr->pred_impl_copy)(xcsf, to, from);
}

static inline void pred_free(XCSF *xcsf, CL *c) {
	(*c->pred_vptr->pred_impl_free)(xcsf, c);
}

static inline void pred_init(XCSF *xcsf, CL *c) {
	(*c->pred_vptr->pred_impl_init)(xcsf, c);
}

static inline void pred_print(XCSF *xcsf, CL *c) {
	(*c->pred_vptr->pred_impl_print)(xcsf, c);
}

static inline void pred_update(XCSF *xcsf, CL *c, double *x, double *y) {
	(*c->pred_vptr->pred_impl_update)(xcsf, c, x, y);
}

// general classifier
_Bool cl_crossover(XCSF *xcsf, CL *c1, CL *c2);
_Bool cl_general(XCSF *xcsf, CL *c1, CL *c2);
_Bool cl_match(XCSF *xcsf, CL *c, double *x);
_Bool cl_match_state(XCSF *xcsf, CL *c);
_Bool cl_mutate(XCSF *xcsf, CL *c);
_Bool cl_subsumer(XCSF *xcsf, CL *c);
double *cl_predict(XCSF *xcsf, CL *c, double *x);
double cl_acc(XCSF *xcsf, CL *c);
double cl_del_vote(XCSF *xcsf, CL *c, double avg_fit);
void cl_copy(XCSF *xcsf, CL *to, CL *from);
void cl_cover(XCSF *xcsf, CL *c, double *x);
void cl_free(XCSF *xcsf, CL *c);
void cl_init(XCSF *xcsf, CL *c, int size, int time);
void cl_print(XCSF *xcsf, CL *c, _Bool print_cond, _Bool print_pred);
void cl_rand(XCSF *xcsf, CL *c);
void cl_update(XCSF *xcsf, CL *c, double *x, double *y, int set_num);
void cl_update_fit(XCSF *xcsf, CL *c, double acc_sum, double acc);

// self-adaptive mutation
double cl_mutation_rate(XCSF *xcsf, CL *c, int m);
void sam_adapt(XCSF *xcsf, double *mu);       
void sam_copy(XCSF *xcsf, double *to, double *from);
void sam_free(XCSF *xcsf, double *mu);
void sam_init(XCSF *xcsf, double **mu);
void sam_print(XCSF *xcsf, double *mu);
