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

struct CondVtbl;
typedef struct CL {
	struct CondVtbl const *cond_vptr; // functions acting on conditions
	struct PredVtbl const *pred_vptr; // functions acting on predictions
	void *cond; // condition structure
	void *pred; // prediction strcuture
	double err;
	double fit;
	int num;
	int exp;
	double size;
	int time;
} CL;

// classifier condition

struct CondVtbl {
	_Bool (*cond_impl_crossover)(CL const * const c1, CL const * const c2);
	_Bool (*cond_impl_general)(CL const * const c1, CL const * const c2);
	_Bool (*cond_impl_match)(CL const * const c, double *x);
	_Bool (*cond_impl_match_state)(CL const * const c);
	_Bool (*cond_impl_mutate)(CL const * const c);
	_Bool (*cond_impl_subsumes)(CL const * const c1, CL const * const c2);
	double (*cond_impl_mu)(CL const * const c, int m);
	void (*cond_impl_copy)(CL const * const to, CL const * const from);
	void (*cond_impl_cover)(CL const * const c, double *x);
	void (*cond_impl_free)(CL const * const c);
	void (*cond_impl_init)(CL const * const c);
	void (*cond_impl_print)(CL const * const c);
	void (*cond_impl_rand)(CL const * const c);
};

static inline _Bool cond_crossover(CL const * const c1, CL const * const c2) {
	return (*c1->cond_vptr->cond_impl_crossover)(c1, c2);
}

static inline _Bool cond_general(CL const * const c1, CL const * const c2) {
	return (*c1->cond_vptr->cond_impl_general)(c1, c2);
}

static inline _Bool cond_match(CL const * const c, double *x) {
	return (*c->cond_vptr->cond_impl_match)(c, x);
}

static inline _Bool cond_match_state(CL const * const c) {
	return (*c->cond_vptr->cond_impl_match_state)(c);
}

static inline _Bool cond_mutate(CL const * const c) {
	return (*c->cond_vptr->cond_impl_mutate)(c);
}

static inline _Bool cond_subsumes(CL const * const c1, CL const * const c2) {
	return (*c1->cond_vptr->cond_impl_subsumes)(c1, c2);
}

static inline double cond_mu(CL const * const c, int m) {
	return (*c->cond_vptr->cond_impl_mu)(c, m);
}

static inline void cond_copy(CL const * const to, CL const * const from) {
	(*to->cond_vptr->cond_impl_copy)(to, from);
}

static inline void cond_cover(CL const * const c, double *x) {
	(*c->cond_vptr->cond_impl_cover)(c, x);
}

static inline void cond_free(CL const * const c) {
	(*c->cond_vptr->cond_impl_free)(c);
}

static inline void cond_init(CL const * const c) {
	(*c->cond_vptr->cond_impl_init)(c);
}

static inline void cond_print(CL const * const c) {
	(*c->cond_vptr->cond_impl_print)(c);
}

static inline void cond_rand(CL const * const c) {
	(*c->cond_vptr->cond_impl_rand)(c);
}

// classifier prediction    

struct PredVtbl {
	double *(*pred_impl_compute)(CL const * const c, double *x);
	double (*pred_impl_pre)(CL const * const c, int p);
	void (*pred_impl_copy)(CL const * const to,  CL const * const from);
	void (*pred_impl_free)(CL const * const c);
	void (*pred_impl_init)(CL const * const c);
	void (*pred_impl_print)(CL const * const c);
	void (*pred_impl_update)(CL const * const c, double *y, double *x);
};
 
static inline double *pred_compute(CL const * const c, double *x) {
	return (*c->pred_vptr->pred_impl_compute)(c, x);
}

static inline double pred_pre(CL const * const c, int p) {
	return (*c->pred_vptr->pred_impl_pre)(c, p);
}

static inline void pred_copy(CL const * const to, CL const * const from) {
	(*to->pred_vptr->pred_impl_copy)(to, from);
}

static inline void pred_free(CL const * const c) {
	(*c->pred_vptr->pred_impl_free)(c);
}

static inline void pred_init(CL const * const c) {
	(*c->pred_vptr->pred_impl_init)(c);
}

static inline void pred_print(CL const * const c) {
	(*c->pred_vptr->pred_impl_print)(c);
}

static inline void pred_update(CL const * const c, double *y, double *x) {
	(*c->pred_vptr->pred_impl_update)(c, y, x);
}
 
// general classifier
_Bool cl_crossover(CL *c1, CL *c2);
_Bool cl_general(CL *c1, CL *c2);
_Bool cl_match(CL *c, double *x);
_Bool cl_match_state(CL *c);
_Bool cl_mutate(CL *c);
_Bool cl_subsumer(CL *c);
_Bool cl_subsumes(CL *c1, CL *c2);
double *cl_predict(CL *c, double *x);
double cl_acc(CL *c);
double cl_del_vote(CL *c, double avg_fit);
void cl_copy(CL *to, CL *from);
void cl_cover(CL *c, double *x);
void cl_free(CL *c);
void cl_init(CL *c, int size, int time);
void cl_print(CL *c);
void cl_rand(CL *c);
void cl_update(CL *c, double *x, double *y, int set_num);
void cl_update_fit(CL *c, double acc_sum, double acc);
 
// self-adaptive mutation
double cl_mutation_rate(CL *c, int m);
void sam_adapt(double *mu);       
void sam_copy(double *to, double *from);
void sam_free(double *mu);
void sam_init(double **mu);
void sam_print(double *mu);
