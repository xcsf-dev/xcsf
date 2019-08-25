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
 
#define CONNECTED 0
#define DROPOUT 1
#define SOFTMAX 2

typedef struct LAYER {
    int layer_type;
    double *output;
    double *weights;
    double *state;
    double *biases;
    double *bias_updates;
    double *weight_updates;
    double *delta;
    int num_inputs;
    int num_outputs;
    int num_weights;
    int activation_type;
    activate_ptr activate;
    gradient_ptr gradient;
    struct LayerVtbl const *layer_vptr; // functions acting on layers
    double scale;
    double probability;
    double *rand;
    double temp;
} LAYER;
 
struct LayerVtbl {
	_Bool (*layer_impl_mutate)(XCSF *xcsf, LAYER *l);
	void (*layer_impl_copy)(XCSF *xcsf, LAYER *to, LAYER *from);
	void (*layer_impl_free)(XCSF *xcsf, LAYER *l);
	void (*layer_impl_rand)(XCSF *xcsf, LAYER *l);
	void (*layer_impl_print)(XCSF *xcsf, LAYER *l, _Bool print_weights);
	void (*layer_impl_update)(XCSF *xcsf, LAYER *l);
	void (*layer_impl_backward)(XCSF *xcsf, LAYER *l, BPN *bpn);
	void (*layer_impl_forward)(XCSF *xcsf, LAYER *l, double *input);
	double* (*layer_impl_output)(XCSF *xcsf, LAYER *l);
};

static inline double* layer_output(XCSF *xcsf, LAYER *l) {
	return (*l->layer_vptr->layer_impl_output)(xcsf, l);
}
    
static inline void layer_forward(XCSF *xcsf, LAYER *l, double *input) {
	(*l->layer_vptr->layer_impl_forward)(xcsf, l, input);
}
  
static inline void layer_backward(XCSF *xcsf, LAYER *l, BPN *bpn) {
	(*l->layer_vptr->layer_impl_backward)(xcsf, l, bpn);
}
 
static inline void layer_update(XCSF *xcsf, LAYER *l) {
	(*l->layer_vptr->layer_impl_update)(xcsf, l);
}

static inline _Bool layer_mutate(XCSF *xcsf, LAYER *l) {
	return (*l->layer_vptr->layer_impl_mutate)(xcsf, l);
}
 
static inline void layer_copy(XCSF *xcsf, LAYER *to, LAYER *from) {
	(*to->layer_vptr->layer_impl_copy)(xcsf, to, from);
}

static inline void layer_free(XCSF *xcsf, LAYER *l) {
	(*l->layer_vptr->layer_impl_free)(xcsf, l);
}

static inline void layer_rand(XCSF *xcsf, LAYER *l) {
	(*l->layer_vptr->layer_impl_rand)(xcsf, l);
}
 
static inline void layer_print(XCSF *xcsf, LAYER *l, _Bool print_weights) {
	(*l->layer_vptr->layer_impl_print)(xcsf, l, print_weights);
}

void neural_layer_init(XCSF *xcsf, LAYER *l, int type, int ninputs, int noutputs, int act);
