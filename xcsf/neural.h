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
    
/**
 * @file neural.h
 * @brief An implementation of a multi-layer perceptron neural network.
 */ 

#pragma once
                     
/**
 * @brief Double linked list of layers data structure.
 */ 
typedef struct LLIST {
    struct LAYER *layer; //!< pointer to the layer data structure
    struct LLIST *prev; //!< pointer to the previous layer (forward)
    struct LLIST *next; //!< pointer to the next layer (backward)
} LLIST;
                      
/**
 * @brief Neural network data structure.
 */  
typedef struct NET {
    int num_layers; //!< number of layers (hidden + output)
    int num_inputs; //!< number of network inputs
    int num_outputs; //!< number of network outputs
    double *delta; //!< delta for updating networks weights
    double *input; //!< pointer to the network input
    LLIST *head; //!< pointer to the head layer (output layer)
    LLIST *tail; //!< pointer to the tail layer (first layer)
} NET;

_Bool neural_mutate(XCSF *xcsf, NET *net);
double neural_output(XCSF *xcsf, NET *net, int i);
int neural_size(XCSF *xcsf, NET *net);
size_t neural_load(XCSF *xcsf, NET *net, FILE *fp);
size_t neural_save(XCSF *xcsf, NET *net, FILE *fp);
void neural_copy(XCSF *xcsf, NET *to, NET *from);
void neural_free(XCSF *xcsf, NET *net);
void neural_init(XCSF *xcsf, NET *net);
void neural_layer_insert(XCSF *xcsf, NET *net, struct LAYER *l, int p);
void neural_layer_remove(XCSF *xcsf, NET *net, int p);
void neural_learn(XCSF *xcsf, NET *net, double *output, double *input);
void neural_print(XCSF *xcsf, NET *net, _Bool print_weights);
void neural_propagate(XCSF *xcsf, NET *net, double *input);
void neural_rand(XCSF *xcsf, NET *net);
