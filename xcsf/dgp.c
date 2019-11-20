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
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "dgp.h"

void graph_init(XCSF *xcsf, GRAPH *dgp, int n)
{
    dgp->t = 0;
    dgp->n = n;
    dgp->klen = n * xcsf->MAX_K;
    dgp->state = malloc(sizeof(double) * dgp->n);
    dgp->initial_state = malloc(sizeof(double) * dgp->n);
    dgp->function = malloc(sizeof(int) * dgp->n);
    dgp->connectivity = malloc(sizeof(int) * dgp->klen);
    dgp->weights = malloc(sizeof(double) * dgp->klen);
}

void graph_copy(XCSF *xcsf, GRAPH *to, GRAPH *from)
{ 	
    (void)xcsf;
    to->t = from->t;
    to->n = from->n;
    to->klen = from->klen;
    memcpy(to->state, from->state, sizeof(double) * from->n);
    memcpy(to->initial_state, from->initial_state, sizeof(double) * from->n);
    memcpy(to->function, from->function, sizeof(int) * from->n);
    memcpy(to->connectivity, from->connectivity, sizeof(int) * from->klen);
    memcpy(to->weights, from->weights, sizeof(double) * from->klen);
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
    dgp->t = irand_uniform(1,xcsf->MAX_T);
    for(int i = 0; i < dgp->n; i++) {
        dgp->function[i] = irand_uniform(0, NUM_ACTIVATIONS);
        dgp->initial_state[i] = rand_normal(0,0.1);
        dgp->state[i] = rand_normal(0,0.1);
    }

    for(int i = 0; i < dgp->klen; i++) {
        dgp->weights[i] = rand_normal(0,0.1);
        // other nodes within the graph
        if(rand_uniform(0,1) < 0.5) {
            dgp->connectivity[i] = irand_uniform(0,dgp->n);
        }
        // external inputs
        else {
            dgp->connectivity[i] = -(irand_uniform(1,xcsf->num_x_vars+1));
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
                int idx = (i * xcsf->MAX_K) + k;
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
            dgp->state[i] = neural_activate(dgp->function[i], dgp->state[i]);
        }
    }
}

void graph_print(XCSF *xcsf, GRAPH *dgp)
{
    printf("Graph: N=%d; T=%d\n", dgp->n, dgp->t);
    for(int i = 0; i < dgp->n; i++) {
        printf("Node %d: func=%s state=%f init_state=%f con=[", 
                i, activation_string(dgp->function[i]),
                dgp->state[i], dgp->initial_state[i]);
        printf("%d", dgp->connectivity[0]);
        for(int j = 1; j < xcsf->MAX_K; j++) {
            printf(",%d", dgp->connectivity[i]);
        }
        printf("]\n");
    }
}

void graph_free(XCSF *xcsf, GRAPH *dgp)
{
    (void)xcsf;
    free(dgp->weights);
    free(dgp->connectivity);
    free(dgp->state);
    free(dgp->initial_state);
    free(dgp->function);
}

_Bool graph_mutate(XCSF *xcsf, GRAPH *dgp)
{
    for(int i = 0; i < dgp->n; i++) {
        // mutate function
        if(rand_uniform(0,1) < xcsf->F_MUTATION) {
            dgp->function[i] = irand_uniform(0, NUM_ACTIVATIONS);
        }
        // mutate initial state
        dgp->initial_state[i] += rand_normal(0, xcsf->S_MUTATION);
        // mutate connectivity map
        for(int j = 0; j < xcsf->MAX_K; j++) {
            int idx = (i*xcsf->MAX_K)+j;
            if(rand_uniform(0,1) < xcsf->P_MUTATION) {
                // external connection
                if(rand_uniform(0,1) < 0.5) {
                    dgp->connectivity[idx] = -(irand_uniform(1,xcsf->num_x_vars+1));
                }
                // another node
                else {
                    dgp->connectivity[idx] = irand_uniform(0,dgp->n);
                }
            }
            // mutate weights
            dgp->weights[idx] += rand_normal(0, xcsf->S_MUTATION);
        }   
    }               
    // mutate T
    if(rand_uniform(0,1) < xcsf->P_MUTATION) {
        if(rand_uniform(0,1) < 0.5) {
            if(dgp->t > 1) {
                (dgp->t)--;
            }
        }
        else {
            if(dgp->t < xcsf->MAX_T) {
                (dgp->t)++;
            }
        }
    }
    return true;
}

_Bool graph_crossover(XCSF *xcsf, GRAPH *dgp1, GRAPH *dgp2)
{
    if(rand_uniform(0,1) > xcsf->P_CROSSOVER) {
        return false;
    }

    if(rand_uniform(0,1) < 0.5) {
        int tmp = dgp1->t;
        dgp1->t = dgp2->t;
        dgp2->t = tmp;
    }

    for(int i = 0; i < dgp1->n; i++) {
        if(rand_uniform(0,1) < 0.5) {
            int tmp = dgp1->function[i];
            dgp1->function[i] = dgp2->function[i];
            dgp2->function[i] = tmp;
        }
        if(rand_uniform(0,1) < 0.5) {
            double tmp = dgp1->initial_state[i];
            dgp1->initial_state[i] = dgp2->initial_state[i];
            dgp2->initial_state[i] = tmp;
        }     
        if(rand_uniform(0,1) < 0.5) {
            double tmp = dgp1->state[i];
            dgp1->state[i] = dgp2->state[i];
            dgp2->state[i] = tmp;
        } 
    }

    for(int i = 0; i < dgp1->klen; i++) {
        if(rand_uniform(0,1) < 0.5) {
            double tmp = dgp1->connectivity[i];
            dgp1->connectivity[i] = dgp2->connectivity[i];
            dgp2->connectivity[i] = tmp;
        }
    }  
 
    for(int i = 0; i < dgp1->klen; i++) {
        if(rand_uniform(0,1) < 0.5) {
            double tmp = dgp1->weights[i];
            dgp1->weights[i] = dgp2->weights[i];
            dgp2->weights[i] = tmp;
        }
    }  
 
    return true;
}

size_t graph_save(XCSF *xcsf, GRAPH *dgp, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fwrite(&dgp->n, sizeof(int), 1, fp);
    s += fwrite(&dgp->t, sizeof(int), 1, fp);
    s += fwrite(&dgp->klen, sizeof(int), 1, fp);
    s += fwrite(dgp->state, sizeof(double), dgp->n, fp);
    s += fwrite(dgp->initial_state, sizeof(double), dgp->n, fp);
    s += fwrite(dgp->function, sizeof(int), dgp->n, fp);
    s += fwrite(dgp->connectivity, sizeof(int), dgp->klen, fp);
    s += fwrite(dgp->weights, sizeof(double), dgp->klen, fp);
    //printf("graph saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t graph_load(XCSF *xcsf, GRAPH *dgp, FILE *fp)
{
    (void)xcsf;
    size_t s = 0;
    s += fread(&dgp->n, sizeof(int), 1, fp);
    s += fread(&dgp->t, sizeof(int), 1, fp);
    s += fread(&dgp->klen, sizeof(int), 1, fp);
    dgp->state = malloc(sizeof(double) * dgp->n);
    dgp->initial_state = malloc(sizeof(double) * dgp->n);
    dgp->function = malloc(sizeof(int) * dgp->n);
    dgp->connectivity = malloc(sizeof(int) * dgp->klen);
    dgp->weights = malloc(sizeof(double) * dgp->klen);
    s += fread(dgp->state, sizeof(double), dgp->n, fp);
    s += fread(dgp->initial_state, sizeof(double), dgp->n, fp);
    s += fread(dgp->function, sizeof(int), dgp->n, fp);
    s += fread(dgp->connectivity, sizeof(int), dgp->klen, fp);
    s += fread(dgp->weights, sizeof(double), dgp->klen, fp);
    //printf("graph loaded %lu elements\n", (unsigned long)s);
    return s;
}
