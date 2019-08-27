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
#include "neural_layer_noise.h"
#include "neural_layer_softmax.h"

void neural_init(XCSF *xcsf, NET *net)
{
    (void)xcsf;
    net->head = NULL;
    net->tail = NULL;
    net->num_layers = 0;
    net->num_inputs = 0;
    net->num_outputs = 0;
}

void neural_layer_add(XCSF *xcsf, NET *net, LAYER *l)
{
    (void)xcsf;
    if(net->head == NULL) {
        net->head = malloc(sizeof(LLIST));
        net->head->layer = l;
        net->head->prev = NULL;
        net->head->next = NULL;
        net->tail = net->head;
        net->num_inputs = l->num_inputs;
        net->num_outputs = l->num_outputs;
    }
    else {
        LLIST *new = malloc(sizeof(LLIST));
        new->layer = l;
        new->next = net->head;
        new->prev = NULL;
        net->head->prev = new;
        net->head = new;
        net->num_outputs = l->num_outputs;
    }
    net->num_layers++;
}

void neural_copy(XCSF *xcsf, NET *to, NET *from)
{
    neural_init(xcsf, to);
    for(LLIST *iter = from->tail; iter != NULL; iter = iter->prev) {
        LAYER *f = iter->layer;
        switch(f->layer_type) {
            case CONNECTED:
                 neural_layer_connected_init(xcsf, to, f->num_inputs, 
                        f->num_outputs, f->activation_type);
                break;
            case DROPOUT:
                neural_layer_dropout_init(xcsf, to, f->num_inputs, f->probability);
                break;
            case NOISE:
                neural_layer_noise_init(xcsf, to, f->num_inputs, f->probability, f->scale);
                break;
            case SOFTMAX:
                neural_layer_softmax_init(xcsf, to, f->num_inputs, f->temp);
                break;
            default:
                printf("neural_copy(): copying from an invalid layer type\n");
                exit(EXIT_FAILURE);
        }
        layer_copy(xcsf, to->head->layer, f);
    }
}

void neural_free(XCSF *xcsf, NET *net)
{
    LLIST *iter = net->tail;
    while(iter != NULL) {
        layer_free(xcsf, iter->layer);
        free(iter->layer);
        net->tail = iter->prev;
        free(iter);
        iter = net->tail;
        net->num_layers--;
    }  
}

void neural_rand(XCSF *xcsf, NET *net)
{
    for(LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        layer_rand(xcsf, iter->layer);
    }
}    

_Bool neural_mutate(XCSF *xcsf, NET *net)
{
    _Bool mod = false;
    for(LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        if(layer_mutate(xcsf, iter->layer)) {
            mod = true;
        }
    }
    return mod;
}

void neural_propagate(XCSF *xcsf, NET *net, double *input)
{
    for(LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        layer_forward(xcsf, iter->layer, input);
        input = layer_output(xcsf, iter->layer);
    }
}

void neural_learn(XCSF *xcsf, NET *net, double *truth, double *input)
{
    /* reset deltas */
    for(LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        LAYER *l = iter->layer;
        for(int j = 0; j < l->num_outputs; j++) {
            l->delta[j] = 0.0;
        }
    }

    // calculate output layer error
    LAYER *p = net->head->layer;
    for(int i = 0; i < p->num_outputs; i++) {
        p->delta[i] = (truth[i] - p->output[i]);
    }

    /* backward phase */
    for(LLIST *iter = net->head; iter != NULL; iter = iter->next) {
        LAYER *l = iter->layer;
        if(iter->next == NULL) {
            net->input = input;
            net->delta = 0;
        }
        else {
            LAYER *prev = iter->next->layer;
            net->input = prev->output;
            net->delta = prev->delta;
        }
        layer_backward(xcsf, l, net);
    }

    /* update phase */
    for(LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        layer_update(xcsf, iter->layer);
    }
} 

double neural_output(XCSF *xcsf, NET *net, int i)
{
    if(i < net->num_outputs) {
        double *output = layer_output(xcsf, net->head->layer);
        return constrain(xcsf->MIN_CON, xcsf->MAX_CON, output[i]);
    }
    printf("neural_output(): requested (%d) in output layer of size (%d)\n",
            i, net->num_outputs);
    exit(EXIT_FAILURE);
}

void neural_print(XCSF *xcsf, NET *net, _Bool print_weights)
{
    int i = 0;
    for(LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        printf("layer (%d) ", i);
        layer_print(xcsf, iter->layer, print_weights);
        i++;
    }
}
