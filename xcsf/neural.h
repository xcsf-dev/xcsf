/*
 * Copyright (C) 2012--2019 Richard Preen <rpreen@gmail.com>
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

// double linked list of layers
typedef struct LLIST {
    struct LAYER *layer;
    struct LLIST *prev;
    struct LLIST *next;
} LLIST;

typedef struct NET {
    int num_layers; // hidden + output
    int num_inputs;
    int num_outputs;
    double *delta;
    double *input;
    LLIST *head;
    LLIST *tail;
} NET;

_Bool neural_crossover(XCSF *xcsf, NET *net1, NET *net2);
_Bool neural_mutate(XCSF *xcsf, NET *net);
double neural_output(XCSF *xcsf, NET *net, int i);
void neural_layer_insert(XCSF *xcsf, NET *net, struct LAYER *l, int p);
void neural_layer_remove(XCSF *xcsf, NET *net, int p);
void neural_copy(XCSF *xcsf, NET *to, NET *from);
void neural_free(XCSF *xcsf, NET *net);
void neural_init(XCSF *xcsf, NET *net);
void neural_learn(XCSF *xcsf, NET *net, double *output, double *input);
void neural_print(XCSF *xcsf, NET *net, _Bool print_weights);
void neural_propagate(XCSF *xcsf, NET *net, double *input);
void neural_rand(XCSF *xcsf, NET *net);
