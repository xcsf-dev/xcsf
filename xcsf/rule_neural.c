/*
 * Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
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
 **************
 * Description: 
 **************
 * The neural classifier rule module.
 * Performs both condition matching and prediction.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "dgp.h"
#include "condition.h"
#include "prediction.h"
#include "rule_neural.h"

typedef struct RULE_NEURAL_COND {
    BPN bpn;
} RULE_NEURAL_COND;

typedef struct RULE_NEURAL_PRED {
    BPN bpn;
    double *input;
} RULE_NEURAL_PRED;

void rule_neural_cond_rand(XCSF *xcsf, CL *c);

void rule_neural_cond_init(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *new = malloc(sizeof(RULE_NEURAL_COND));
    // initialise empty network
    neural_init(xcsf, &new->bpn);
    // create and add layers to the network
    neural_layer_connected_init(xcsf, &new->bpn,
            xcsf->num_x_vars, xcsf->NUM_HIDDEN_NEURONS, xcsf->HIDDEN_NEURON_ACTIVATION);
    neural_layer_connected_init(xcsf, &new->bpn,
            xcsf->NUM_HIDDEN_NEURONS, xcsf->MAX_FORWARD+1, IDENTITY);
    // initialise all weights randomly
    neural_rand(xcsf, &new->bpn);
    c->cond = new;
}

void rule_neural_cond_free(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = c->cond;
    neural_free(xcsf, &cond->bpn);
    free(c->cond);
}  

void rule_neural_cond_copy(XCSF *xcsf, CL *to, CL *from)
{
    RULE_NEURAL_COND *new = malloc(sizeof(RULE_NEURAL_COND));
    RULE_NEURAL_COND *from_cond = from->cond;
    neural_copy(xcsf, &new->bpn, &from_cond->bpn);
    to->cond = new;
}

void rule_neural_cond_rand(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = c->cond;
    neural_rand(xcsf, &cond->bpn);
}

void rule_neural_cond_cover(XCSF *xcsf, CL *c, double *x)
{
    do {
        rule_neural_cond_rand(xcsf, c);
    } while(!rule_neural_cond_match(xcsf, c, x));
}
 
void rule_neural_cond_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}
 
_Bool rule_neural_cond_match(XCSF *xcsf, CL *c, double *x)
{
    RULE_NEURAL_COND *cond = c->cond;
    neural_propagate(xcsf, &cond->bpn, x);
    if(neural_output(xcsf, &cond->bpn, 0) > 0.5) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}    

_Bool rule_neural_cond_mutate(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = c->cond;
    return neural_mutate(xcsf, &cond->bpn);
}

_Bool rule_neural_cond_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_neural_cond_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}   

void rule_neural_cond_print(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = c->cond;
    neural_print(xcsf, &cond->bpn, true);
}  

void rule_neural_pred_init(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_PRED *new = malloc(sizeof(RULE_NEURAL_PRED));
    // initialise empty network
    neural_init(xcsf, &new->bpn);
    // create and add layers to the network
    neural_layer_connected_init(xcsf, &new->bpn,
            xcsf->MAX_FORWARD, xcsf->NUM_HIDDEN_NEURONS, xcsf->HIDDEN_NEURON_ACTIVATION);
    neural_layer_connected_init(xcsf, &new->bpn, 
            xcsf->NUM_HIDDEN_NEURONS, xcsf->num_y_vars, IDENTITY);
    // initialise all weights randomly
    neural_rand(xcsf, &new->bpn);
    new->input = malloc(sizeof(double)*xcsf->MAX_FORWARD);
    c->pred = new;  
}

void rule_neural_pred_free(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_PRED *pred = c->pred;
    neural_free(xcsf, &pred->bpn);
    free(pred->input);
    free(c->pred);
}

void rule_neural_pred_copy(XCSF *xcsf, CL *to, CL *from)
{
    RULE_NEURAL_PRED *new = malloc(sizeof(RULE_NEURAL_PRED));
    RULE_NEURAL_PRED *from_pred = from->pred;
    neural_copy(xcsf, &new->bpn, &from_pred->bpn);
    new->input = malloc(sizeof(double)*xcsf->MAX_FORWARD);
    to->pred = new;
}

void rule_neural_pred_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)x;
    RULE_NEURAL_PRED *pred = c->pred;
    neural_learn(xcsf, &pred->bpn, y, pred->input);
}

double *rule_neural_pred_compute(XCSF *xcsf, CL *c, double *x)
{
    (void)x;
    RULE_NEURAL_COND *cond = c->cond;
    RULE_NEURAL_PRED *pred = c->pred;
    for(int i = 0; i < xcsf->MAX_FORWARD; i++) {
        pred->input[i] = neural_output(xcsf, &cond->bpn, 1+i);
    }
    neural_propagate(xcsf, &pred->bpn, pred->input);
    for(int i = 0; i <  xcsf->num_y_vars; i++) {
        c->prediction[i] = neural_output(xcsf, &pred->bpn, i);
    }
    return c->prediction;
}

void rule_neural_pred_print(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_PRED *pred = c->pred;
    neural_print(xcsf, &pred->bpn, true);
}  

_Bool rule_neural_pred_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_neural_pred_mutate(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
    return false;
}
