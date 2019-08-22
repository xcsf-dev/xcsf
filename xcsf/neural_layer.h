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

struct LayerVtbl {
	void (*layer_impl_init)(LAYER *l, int num_inputs, int num_outputs, int activation);
	_Bool (*layer_impl_mutate)(XCSF *xcsf, LAYER *l);
	void (*layer_impl_copy)(LAYER *to,  LAYER *from);
	void (*layer_impl_free)(LAYER *l);
	void (*layer_impl_rand)(LAYER *l);
	void (*layer_impl_print)(LAYER *l, _Bool print_weights);
	void (*layer_impl_update)(XCSF *xcsf, LAYER *l);
	void (*layer_impl_backward)(LAYER *l);
	void (*layer_impl_forward)(LAYER *l, double *input);
	double* (*layer_impl_output)(LAYER *l);
};

static inline double* layer_output(LAYER *l) {
	return (*l->layer_vptr->layer_impl_output)(l);
}
    
static inline void layer_init(LAYER *l, int num_inputs, int num_outputs, int activation) {
	(*l->layer_vptr->layer_impl_init)(l, num_inputs, num_outputs, activation);
}
   
static inline void layer_forward(LAYER *l, double *input) {
	(*l->layer_vptr->layer_impl_forward)(l, input);
}
  
static inline void layer_backward(LAYER *l) {
	(*l->layer_vptr->layer_impl_backward)(l);
}
 
static inline void layer_update(XCSF *xcsf, LAYER *l) {
	(*l->layer_vptr->layer_impl_update)(xcsf, l);
}

static inline _Bool layer_mutate(XCSF *xcsf, LAYER *l) {
	return (*l->layer_vptr->layer_impl_mutate)(xcsf, l);
}
 
static inline void layer_copy(LAYER *to, LAYER *from) {
	(*to->layer_vptr->layer_impl_copy)(to, from);
}

static inline void layer_free(LAYER *l) {
	(*l->layer_vptr->layer_impl_free)(l);
}

static inline void layer_rand(LAYER *l) {
	(*l->layer_vptr->layer_impl_rand)(l);
}
 
static inline void layer_print(LAYER *l, _Bool print_weights) {
	(*l->layer_vptr->layer_impl_print)(l, print_weights);
}
