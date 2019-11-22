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

     
/**
 * @file action.h
 * @brief Interface for classifier actions.
 */ 
 
#pragma once

void action_set(XCSF *xcsf, CL *c);

/**
 * @brief Action interface data structure.
 */ 
struct ActVtbl {
	_Bool (*act_impl_general)(XCSF *xcsf, CL *c1, CL *c2);
	_Bool (*act_impl_crossover)(XCSF *xcsf, CL *c1, CL *c2);
	_Bool (*act_impl_mutate)(XCSF *xcsf, CL *c);
	double *(*act_impl_compute)(XCSF *xcsf, CL *c, double *x);
	void (*act_impl_copy)(XCSF *xcsf, CL *to,  CL *from);
	void (*act_impl_free)(XCSF *xcsf, CL *c);
	void (*act_impl_init)(XCSF *xcsf, CL *c);
	void (*act_impl_rand)(XCSF *xcsf, CL *c);
	void (*act_impl_print)(XCSF *xcsf, CL *c);
	void (*act_impl_update)(XCSF *xcsf, CL *c, double *x, double *y);
	size_t (*act_impl_save)(XCSF *xcsf, CL *c, FILE *fp);
	size_t (*act_impl_load)(XCSF *xcsf, CL *c, FILE *fp);
};

static inline size_t act_save(XCSF *xcsf, CL *c, FILE *fp) {
	return (*c->act_vptr->act_impl_save)(xcsf, c, fp);
}
 
static inline size_t act_load(XCSF *xcsf, CL *c, FILE *fp) {
	return (*c->act_vptr->act_impl_load)(xcsf, c, fp);
}
 
static inline _Bool act_general(XCSF *xcsf, CL *c1, CL *c2) {
	return (*c1->act_vptr->act_impl_general)(xcsf, c1, c2);
}
 
static inline _Bool act_crossover(XCSF *xcsf, CL *c1, CL *c2) {
	return (*c1->act_vptr->act_impl_crossover)(xcsf, c1, c2);
}
 
static inline _Bool act_mutate(XCSF *xcsf, CL *c) {
	return (*c->act_vptr->act_impl_mutate)(xcsf, c);
}
 
static inline double *act_compute(XCSF *xcsf, CL *c, double *x) {
	return (*c->act_vptr->act_impl_compute)(xcsf, c, x);
}

static inline void act_copy(XCSF *xcsf, CL *to, CL *from) {
	(*from->act_vptr->act_impl_copy)(xcsf, to, from);
}

static inline void act_free(XCSF *xcsf, CL *c) {
	(*c->act_vptr->act_impl_free)(xcsf, c);
}

static inline void act_init(XCSF *xcsf, CL *c) {
	(*c->act_vptr->act_impl_init)(xcsf, c);
}
 
static inline void act_rand(XCSF *xcsf, CL *c) {
	(*c->act_vptr->act_impl_rand)(xcsf, c);
}
 
static inline void act_print(XCSF *xcsf, CL *c) {
	(*c->act_vptr->act_impl_print)(xcsf, c);
}

static inline void act_update(XCSF *xcsf, CL *c, double *x, double *y) {
	(*c->act_vptr->act_impl_update)(xcsf, c, x, y);
}
