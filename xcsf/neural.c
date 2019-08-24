/*
 * Copyright (C) 2016--2019 Richard Preen <rpreen@gmail.com>
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
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"

void neural_add_layer(XCSF *xcsf, BPN *bpn, LAYER *l);

void neural_init(XCSF *xcsf, BPN *bpn, int num_layers, int *neurons, int *activations)
{
    bpn->head = NULL;
    bpn->tail = NULL;
    bpn->num_layers = 0;
    bpn->num_inputs = neurons[0];
    bpn->num_outputs = neurons[num_layers-1];
    for(int i = 0; i < num_layers-1; i++) {
        LAYER *l = neural_layer_connected_init(xcsf, neurons[i], neurons[i+1], activations[i]);
        neural_add_layer(xcsf, bpn, l);
    }

    //LAYER *l;
    //l = neural_layer_connected_init(xcsf, neurons[0], neurons[1], activations[0]);
    //neural_add_layer(xcsf, bpn, l);
    //l = neural_layer_dropout_init(xcsf, neurons[1], 0.1);
    //neural_add_layer(xcsf, bpn, l);
    //l = neural_layer_connected_init(xcsf, neurons[1], neurons[2], activations[1]);
    //neural_add_layer(xcsf, bpn, l);
}

void neural_add_layer(XCSF *xcsf, BPN *bpn, LAYER *l)
{
    (void)xcsf;
    if(bpn->head == NULL) {
        bpn->head = malloc(sizeof(LLIST));
        bpn->head->layer = l;
        bpn->head->prev = NULL;
        bpn->head->next = NULL;
        bpn->tail = bpn->head;
    }
    else {
        LLIST *new = malloc(sizeof(LLIST));
        new->layer = l;
        new->next = bpn->head;
        new->prev = NULL;
        bpn->head->prev = new;
        bpn->head = new;
    }
    bpn->num_layers++;
}

void neural_copy(XCSF *xcsf, BPN *to, BPN *from)
{
    to->head = NULL;
    to->tail = NULL;
    to->num_layers = 0;
    to->num_outputs = from->num_outputs;
    to->num_inputs = from->num_inputs;
    for(LLIST *iter = from->tail; iter != NULL; iter = iter->prev) {
        LAYER *new;
        LAYER *f = iter->layer;
        switch(f->layer_type) {
            case CONNECTED:
                new = neural_layer_connected_init(xcsf, f->num_inputs, 
                        f->num_outputs, f->activation_type);
                break;
            case DROPOUT:
                new = neural_layer_dropout_init(xcsf, f->num_inputs, f->probability);
                break;
            default:
                printf("neural_copy(): copying from an invalid layer type\n");
                exit(EXIT_FAILURE);
        }
        layer_copy(xcsf, new, f);
        neural_add_layer(xcsf, to, new);
    }
}

void neural_free(XCSF *xcsf, BPN *bpn)
{
    LLIST *iter = bpn->tail;
    while(iter != NULL) {
        layer_free(xcsf, iter->layer);
        free(iter->layer);
        bpn->tail = iter->prev;
        free(iter);
        iter = bpn->tail;
        bpn->num_layers--;
    }  
}

void neural_rand(XCSF *xcsf, BPN *bpn)
{
    for(LLIST *iter = bpn->tail; iter != NULL; iter = iter->prev) {
        layer_rand(xcsf, iter->layer);
    }
}    

_Bool neural_mutate(XCSF *xcsf, BPN *bpn)
{
    _Bool mod = false;
    for(LLIST *iter = bpn->tail; iter != NULL; iter = iter->prev) {
        if(layer_mutate(xcsf, iter->layer)) {
            mod = true;
        }
    }
    return mod;
}

void neural_propagate(XCSF *xcsf, BPN *bpn, double *input)
{
    for(LLIST *iter = bpn->tail; iter != NULL; iter = iter->prev) {
        layer_forward(xcsf, iter->layer, input);
        input = layer_output(xcsf, iter->layer);
    }
}

void neural_learn(XCSF *xcsf, BPN *bpn, double *truth, double *input)
{
    /* reset deltas */
    for(LLIST *iter = bpn->tail; iter != NULL; iter = iter->prev) {
        LAYER *l = iter->layer;
        for(int j = 0; j < l->num_outputs; j++) {
            l->delta[j] = 0.0;
        }
    }

    // calculate output layer error
    LAYER *p = bpn->head->layer;
    for(int i = 0; i < p->num_outputs; i++) {
        p->delta[i] = (truth[i] - p->output[i]);
    }

    /* backward phase */
    for(LLIST *iter = bpn->head; iter != NULL; iter = iter->next) {
        LAYER *l = iter->layer;
        if(iter->next == NULL) {
            bpn->input = input;
            bpn->delta = 0;
        }
        else {
            LAYER *prev = iter->next->layer;
            bpn->input = prev->output;
            bpn->delta = prev->delta;
        }
        layer_backward(xcsf, l, bpn);
    }

    /* update phase */
    for(LLIST *iter = bpn->tail; iter != NULL; iter = iter->prev) {
        layer_update(xcsf, iter->layer);
    }

} 

double neural_output(XCSF *xcsf, BPN *bpn, int i)
{
    if(i < bpn->num_outputs) {
        double *output = layer_output(xcsf, bpn->head->layer);
        return constrain(xcsf->MIN_CON, xcsf->MAX_CON, output[i]);
    }
    printf("neural_output(): requested (%d) in output layer of size (%d)\n",
            i, bpn->num_outputs);
    exit(EXIT_FAILURE);
}

void neural_print(XCSF *xcsf, BPN *bpn, _Bool print_weights)
{
    int i = 0;
    for(LLIST *iter = bpn->tail; iter != NULL; iter = iter->prev) {
        printf("layer (%d) ", i);
        layer_print(xcsf, iter->layer, print_weights);
        i++;
    }
}
