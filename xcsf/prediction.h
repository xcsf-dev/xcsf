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

void prediction_set(XCSF *xcsf, CL *c);

struct PredVtbl {
	_Bool (*pred_impl_crossover)(XCSF *xcsf, CL *c1, CL *c2);
	_Bool (*pred_impl_mutate)(XCSF *xcsf, CL *c);
	double *(*pred_impl_compute)(XCSF *xcsf, CL *c, double *x);
	void (*pred_impl_copy)(XCSF *xcsf, CL *to,  CL *from);
	void (*pred_impl_free)(XCSF *xcsf, CL *c);
	void (*pred_impl_init)(XCSF *xcsf, CL *c);
	void (*pred_impl_print)(XCSF *xcsf, CL *c);
	void (*pred_impl_update)(XCSF *xcsf, CL *c, double *x, double *y);
};
 
static inline _Bool pred_crossover(XCSF *xcsf, CL *c1, CL *c2) {
	return (*c1->pred_vptr->pred_impl_crossover)(xcsf, c1, c2);
}
 
static inline _Bool pred_mutate(XCSF *xcsf, CL *c) {
	return (*c->pred_vptr->pred_impl_mutate)(xcsf, c);
}
 
static inline double *pred_compute(XCSF *xcsf, CL *c, double *x) {
	return (*c->pred_vptr->pred_impl_compute)(xcsf, c, x);
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
