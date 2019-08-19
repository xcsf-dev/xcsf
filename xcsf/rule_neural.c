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
#include "data_structures.h"
#include "random.h"
#include "cl.h"
#include "neural.h"
#include "dgp.h"
#include "rule_neural.h"

typedef struct RULE_NEURAL_COND {
    BPN bpn;
    _Bool m;
    double *mu;
} RULE_NEURAL_COND;

typedef struct RULE_NEURAL_PRED {
    BPN bpn;
    double *input;
    double *pre;
} RULE_NEURAL_PRED;

void rule_neural_cond_init(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = malloc(sizeof(RULE_NEURAL_COND));
    // network with 1 hidden layer
    int neurons[3] = {xcsf->num_x_vars, xcsf->NUM_HIDDEN_NEURONS, 1+xcsf->MAX_FORWARD};
    // select layer activation functions
    int activations[2] = {xcsf->HIDDEN_NEURON_ACTIVATION, IDENTITY};
    // initialise neural network
    neural_init(xcsf, &cond->bpn, 3, neurons, activations);
    c->cond = cond;
    sam_init(xcsf, &cond->mu);
}

void rule_neural_cond_free(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = c->cond;
    neural_free(xcsf, &cond->bpn);
    sam_free(xcsf, cond->mu);
    free(c->cond);
}  

double rule_neural_cond_mu(XCSF *xcsf, CL *c, int m)
{
    (void)xcsf;
    RULE_NEURAL_COND *cond = c->cond;
    return cond->mu[m];
}

void rule_neural_cond_copy(XCSF *xcsf, CL *to, CL *from)
{
    RULE_NEURAL_COND *to_cond = to->cond;
    RULE_NEURAL_COND *from_cond = from->cond;
    neural_copy(xcsf, &to_cond->bpn, &from_cond->bpn);
    sam_copy(xcsf, to_cond->mu, from_cond->mu);
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

_Bool rule_neural_cond_match(XCSF *xcsf, CL *c, double *x)
{
    // classifier matches if the first output neuron > 0.5
    RULE_NEURAL_COND *cond = c->cond;
    neural_propagate(xcsf, &cond->bpn, x);
    if(neural_output(xcsf, &cond->bpn, 0) > 0.5) {
        cond->m = true;
    }
    else {
        cond->m = false;
    }
    return cond->m;
}    

_Bool rule_neural_cond_match_state(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    RULE_NEURAL_COND *cond = c->cond;
    return cond->m;
}

_Bool rule_neural_cond_mutate(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = c->cond;
    // update mutation rates
    sam_adapt(xcsf, cond->mu);
    // apply mutation
    return neural_mutate(xcsf, &cond->bpn);
}

_Bool rule_neural_cond_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}

_Bool rule_neural_cond_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf;
    (void)c1;
    (void)c2;
    return false;
}   

void rule_neural_cond_print(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_COND *cond = c->cond;
    neural_print(xcsf, &cond->bpn);
}  

void rule_neural_pred_init(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_PRED *pred = malloc(sizeof(RULE_NEURAL_COND));
    int neurons[3] = {xcsf->MAX_FORWARD, xcsf->NUM_HIDDEN_NEURONS, xcsf->num_y_vars};
    // select layer activation functions
    int activations[2] = {xcsf->HIDDEN_NEURON_ACTIVATION, IDENTITY};
    // initialise neural network
    neural_init(xcsf, &pred->bpn, 3, neurons, activations);
    pred->pre = malloc(sizeof(double) * xcsf->num_y_vars);
    pred->input = malloc(sizeof(double) * xcsf->MAX_FORWARD);
    c->pred = pred;
}

void rule_neural_pred_free(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_PRED *pred = c->pred;
    neural_free(xcsf, &pred->bpn);
    free(pred->pre);
    free(pred->input);
    free(c->pred);
}

void rule_neural_pred_copy(XCSF *xcsf, CL *to, CL *from)
{
    RULE_NEURAL_PRED *to_pred = to->pred;
    RULE_NEURAL_PRED *from_pred = from->pred;
    neural_copy(xcsf, &to_pred->bpn, &from_pred->bpn);
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
    // get output of condition network (propagated during matching)
    RULE_NEURAL_COND *cond = c->cond;
    RULE_NEURAL_PRED *pred = c->pred;
    for(int i = 0; i < xcsf->MAX_FORWARD; i++) {
        pred->input[i] = neural_output(xcsf, &cond->bpn, 1+i);
    }
    // propagate outputs through prediction network
    neural_propagate(xcsf, &pred->bpn, pred->input);
    for(int i = 0; i <  xcsf->num_y_vars; i++) {
        pred->pre[i] = neural_output(xcsf, &pred->bpn, i);
    }
    return pred->pre;
}

double *rule_neural_pred_pre(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    RULE_NEURAL_PRED *pred = c->pred;
    return pred->pre;
}

void rule_neural_pred_print(XCSF *xcsf, CL *c)
{
    RULE_NEURAL_PRED *pred = c->pred;
    neural_print(xcsf, &pred->bpn);
}  
