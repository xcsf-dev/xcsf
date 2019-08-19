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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "data_structures.h"
#include "random.h"
#include "neural.h"
#include "dgp.h"

void node_set_activation(activ_ptr *activ, int func);

void graph_init(XCSF *xcsf, GRAPH *dgp, int n)
{
    dgp->t = 0;
    dgp->n = n;
    dgp->state = malloc(sizeof(double)*dgp->n);
    dgp->initial_state = malloc(sizeof(double)*dgp->n);
    dgp->activ = malloc(sizeof(activ_ptr)*dgp->n);
    dgp->connectivity = malloc(sizeof(int)*dgp->n*xcsf->MAX_K);
    dgp->weights = malloc(sizeof(double)*dgp->n*xcsf->MAX_K);
    graph_rand(xcsf, dgp);
}

void graph_copy(XCSF *xcsf, GRAPH *to, GRAPH *from)
{ 	
    to->t = from->t;
    to->n = from->n;
    memcpy(to->state, from->state, sizeof(double)*from->n);
    memcpy(to->initial_state, from->initial_state, sizeof(double)*from->n);
    memcpy(to->activ, from->activ, sizeof(activ_ptr)*from->n);
    memcpy(to->connectivity, from->connectivity, sizeof(int)*from->n*xcsf->MAX_K);
    memcpy(to->weights, from->weights, sizeof(double)*from->n*xcsf->MAX_K);
}

double graph_output(XCSF *xcsf, GRAPH *dgp, int i)
{
    (void)xcsf;
    return dgp->state[i];
}

void graph_reset(XCSF *xcsf, GRAPH *dgp)
{
    (void)xcsf;
    for(int i = 0; i < dgp->n; i++) {
        dgp->state[i] = dgp->initial_state[i];
    }
}

void graph_rand(XCSF *xcsf, GRAPH *dgp)
{
    dgp->t = irand(1,xcsf->MAX_T);
    for(int i = 0; i < dgp->n; i++) {
        node_set_activation(&dgp->activ[i], irand(0, NUM_ACTIVATIONS));
        dgp->initial_state[i] = ((xcsf->MAX_CON-xcsf->MIN_CON)*drand())+xcsf->MIN_CON;
        dgp->state[i] = ((xcsf->MAX_CON-xcsf->MIN_CON)*drand())+xcsf->MIN_CON;
    }

    for(int i = 0; i < dgp->n * xcsf->MAX_K; i++) {
        dgp->weights[i] = (drand()*2.0)-1.0;
        // other nodes within the graph
        if(drand() < 0.5) {
            dgp->connectivity[i] = irand(0,dgp->n);
        }
        // external inputs
        else {
            dgp->connectivity[i] = -(irand(1,xcsf->num_x_vars+1));
        }
    }
}

void graph_update(XCSF *xcsf, GRAPH *dgp, double *inputs)
{
    if(xcsf->RESET_STATES) {
        graph_reset(xcsf, dgp);
    }
    for(int t = 0; t < dgp->t; t++) {
        // synchronously update each node
        for(int i = dgp->n-1; i >= 0; i--) {
            // each connection
            for(int k = 0; k < xcsf->MAX_K; k++) {
                int idx = (i*xcsf->MAX_K)+k;
                int c = dgp->connectivity[idx];
                // another node within the graph
                if(c >= 0) {
                    dgp->state[i] += dgp->weights[idx] * dgp->state[c];
                }
                // external input
                else {
                    dgp->state[i] += dgp->weights[idx] * inputs[abs(c)-1];
                }
            }
            dgp->state[i] = dgp->activ[i](dgp->state[i]);
        }
    }
}

void graph_print(XCSF *xcsf, GRAPH *dgp)
{
    printf("Graph: N=%d; T=%d\n", dgp->n, dgp->t);
    for(int i = 0; i < dgp->n; i++) {
        printf("Node %d: state=%f init_state=%f con=[", 
                i, dgp->state[i], dgp->initial_state[i]);
        printf("%d", dgp->connectivity[0]);
        for(int j = 1; j < xcsf->MAX_K; j++) {
            printf(", %d", dgp->connectivity[i]);
        }
        printf("]\n");
    }
}

void graph_free(XCSF *xcsf, GRAPH *dgp)
{
    free(dgp->weights);
    free(dgp->connectivity);
    free(dgp->state);
    free(dgp->initial_state);
    free(dgp->activ);
    (void)xcsf;
}

_Bool graph_mutate(XCSF *xcsf, GRAPH *dgp)
{
    _Bool fmodified = false;
    _Bool cmodified = false;
    _Bool tmodified = false;

    for(int i = 0; i < dgp->n; i++) {
        // mutate function
        if(drand() < xcsf->P_FUNC_MUTATION) {
            activ_ptr old = dgp->activ[i];
            node_set_activation(&dgp->activ[i], irand(0, NUM_ACTIVATIONS));
            if(old != dgp->activ[i]) {
                fmodified = true;
            }              
        }
        // mutate initial state
        if(drand() < xcsf->P_MUTATION) {
            dgp->initial_state[i] += ((drand()*2.0)-1.0)*xcsf->S_MUTATION;
        }

        // mutate connectivity map
        for(int j = 0; j < xcsf->MAX_K; j++) {
            int idx = (i*xcsf->MAX_K)+j;
            if(drand() < xcsf->P_MUTATION) {
                int old = dgp->connectivity[idx];
                // external connection
                if(drand() < 0.5) {
                    dgp->connectivity[idx] = -(irand(1,xcsf->num_x_vars+1));
                }
                // another node
                else {
                    dgp->connectivity[idx] = irand(0,dgp->n);
                }
                if(old != dgp->connectivity[idx]) {
                    cmodified = true;
                }
            }
            // mutate weights
            if(drand() < xcsf->P_MUTATION) {
                dgp->weights[idx] += ((drand()*2.0)-1.0)*xcsf->S_MUTATION;
            }
        }   
    }               

    // mutate T
    if(drand() < xcsf->P_MUTATION) {
        int t = dgp->t;
        if(drand() < 0.5) {
            if(dgp->t > 1) {
                (dgp->t)--;
            }
        }
        else {
            if(dgp->t < xcsf->MAX_T) {
                (dgp->t)++;
            }
        }
        if(t != dgp->t) {
            tmodified = true;
        }
    }

    if(fmodified || cmodified || tmodified) {
        return true;
    }
    else {
        return false;
    }
}

_Bool graph_crossover(XCSF *xcsf, GRAPH *dgp1, GRAPH *dgp2)
{
    // uniform crossover -- due to the competing conventions problem
    // P_CROSSOVER = 0.0 may perform better
    if(drand() > xcsf->P_CROSSOVER) {
        return false;
    }

    // cross number of cycles
    if(drand() < 0.5) {
        int tmp = dgp1->t;
        dgp1->t = dgp2->t;
        dgp2->t = tmp;
    }

    // cross functions and states
    for(int i = 0; i < dgp1->n; i++) {
        if(drand() < 0.5) {
            activ_ptr tmp = dgp1->activ[i];
            dgp1->activ[i] = dgp2->activ[i];
            dgp2->activ[i] = tmp;
        }
        if(drand() < 0.5) {
            double tmp = dgp1->initial_state[i];
            dgp1->initial_state[i] = dgp2->initial_state[i];
            dgp2->initial_state[i] = tmp;
        }     
        if(drand() < 0.5) {
            double tmp = dgp1->state[i];
            dgp1->state[i] = dgp2->state[i];
            dgp2->state[i] = tmp;
        } 
    }

    // cross connections
    for(int i = 0; i < dgp1->n * xcsf->MAX_K; i++) {
        if(drand() < 0.5) {
            double tmp = dgp1->connectivity[i];
            dgp1->connectivity[i] = dgp2->connectivity[i];
            dgp2->connectivity[i] = tmp;
        }
    }  
 
    // cross weights
    for(int i = 0; i < dgp1->n * xcsf->MAX_K; i++) {
        if(drand() < 0.5) {
            double tmp = dgp1->weights[i];
            dgp1->weights[i] = dgp2->weights[i];
            dgp2->weights[i] = tmp;
        }
    }  
 
    return true;
}

void node_set_activation(activ_ptr *activ, int func)
{
    switch(func) {
        case LOGISTIC:
            *activ = &logistic_activ;
            break;
        case RELU:
            *activ = &relu_activ;
            break;
        case GAUSSIAN:
            *activ = &gaussian_activ;
            break;
        case BENT_IDENTITY:
            *activ = &bent_identity_activ;
            break;
        case TANH:
            *activ = &tanh_activ;
            break;
        case SIN:
            *activ = &sin;
            break;
        case COS:
            *activ = &cos;
            break;
        case SOFT_PLUS:
            *activ = &soft_plus_activ;
            break;
        case IDENTITY:
            *activ = &identity_activ;
            break;
        case HARDTAN:
            *activ = &hardtan_activ;
            break;
        case STAIR:
            *activ = &stair_activ;
            break;
        case LEAKY:
            *activ = &leaky_activ;
            break;
        case ELU:
            *activ = &elu_activ;
            break;
        case RAMP:
            *activ = &ramp_activ;
            break;
        default:
            printf("error: invalid activation function: %d\n", func);
            exit(EXIT_FAILURE);
    }                                    
}
