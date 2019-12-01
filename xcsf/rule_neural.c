/*
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
 * @file rule_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019.
 * @brief Neural network rule (condition + action) functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "cl.h"
#include "condition.h"
#include "action.h"
#include "rule_neural.h"
 
/**
 * @brief Neural network rule data structure.
 */ 
typedef struct RULE_NEURAL {
    NET net; //!< Neural network
    int num_outputs; //!< Number of action nodes (binarised)
} RULE_NEURAL;

/* CONDITION FUNCTIONS */

void rule_neural_cond_rand(XCSF *xcsf, CL *c);

void rule_neural_cond_init(XCSF *xcsf, CL *c)
{
    RULE_NEURAL *new = malloc(sizeof(RULE_NEURAL));
    neural_init(xcsf, &new->net);
    // weights
    uint32_t lopt = 0;
    if(xcsf->COND_EVOLVE_WEIGHTS) {
        lopt |= LAYER_EVOLVE_WEIGHTS;
    }
    // neurons
    int hmax = fmax(xcsf->COND_MAX_HIDDEN_NEURONS, 1);
    int hinit = xcsf->COND_NUM_HIDDEN_NEURONS;
    if(hinit < 1) {
        hinit = irand_uniform(1, hmax);
    }
    if(hmax < hinit) {
        hmax = hinit;
    }
    if(xcsf->COND_EVOLVE_NEURONS) {
        lopt |= LAYER_EVOLVE_NEURONS;
    }
    else {
        hmax = hinit;
    }
    // functions
    int f = xcsf->COND_HIDDEN_NEURON_ACTIVATION;
    if(xcsf->COND_EVOLVE_FUNCTIONS) {
        lopt |= LAYER_EVOLVE_FUNCTIONS;
    }
    // hidden layer
    LAYER *l = neural_layer_connected_init(xcsf, xcsf->num_x_vars, hinit, hmax, f, lopt);
    neural_layer_insert(xcsf, &new->net, l, 0); 
    // output layer
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    int n = fmax(1, ceil(log2(xcsf->num_actions))); // number of action neurons
    new->num_outputs = n;
    l = neural_layer_connected_init(xcsf, hmax, n+1, n+1, LOGISTIC, lopt);
    neural_layer_insert(xcsf, &new->net, l, 1); 
    c->cond = new; 
}

void rule_neural_cond_free(XCSF *xcsf, CL *c)
{
    RULE_NEURAL *cond = c->cond;
    neural_free(xcsf, &cond->net);
    free(c->cond);
}

void rule_neural_cond_copy(XCSF *xcsf, CL *to, CL *from)
{
    RULE_NEURAL *new = malloc(sizeof(RULE_NEURAL));
    RULE_NEURAL *from_cond = from->cond;
    new->num_outputs = from_cond->num_outputs;
    neural_copy(xcsf, &new->net, &from_cond->net);
    to->cond = new;
}

void rule_neural_cond_rand(XCSF *xcsf, CL *c)
{
    RULE_NEURAL *cond = c->cond;
    neural_rand(xcsf, &cond->net);
}

void rule_neural_cond_cover(XCSF *xcsf, CL *c, double *x)
{
    (void)xcsf; (void)c; (void)x;
}

void rule_neural_cond_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool rule_neural_cond_match(XCSF *xcsf, CL *c, double *x)
{
    RULE_NEURAL *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if(neural_output(xcsf, &cond->net, 0) > 0.5) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}    

_Bool rule_neural_cond_mutate(XCSF *xcsf, CL *c)
{
    RULE_NEURAL *cond = c->cond;
    return neural_mutate(xcsf, &cond->net);
}

_Bool rule_neural_cond_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void) c1; (void)c2;
    return false;
}

_Bool rule_neural_cond_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}   

void rule_neural_cond_print(XCSF *xcsf, CL *c)
{
    RULE_NEURAL *cond = c->cond;
    neural_print(xcsf, &cond->net, false);
}  
 
int rule_neural_cond_size(XCSF *xcsf, CL *c)
{
    RULE_NEURAL *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t rule_neural_cond_save(XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_NEURAL *cond = c->cond;
    size_t s = neural_save(xcsf, &cond->net, fp);
    //printf("rule neural saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t rule_neural_cond_load(XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_NEURAL *new = malloc(sizeof(RULE_NEURAL));
    size_t s = neural_load(xcsf, &new->net, fp);
    new->num_outputs = fmax(1, ceil(log2(xcsf->num_actions)));
    c->cond = new;
    //printf("rule neural loaded %lu elements\n", (unsigned long)s);
    return s;
}

/* ACTION FUNCTIONS */

void rule_neural_act_init(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void rule_neural_act_free(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}
 
void rule_neural_act_copy(XCSF *xcsf, CL *to, CL *from)
{
    (void)xcsf; (void)to; (void)from;
}
 
void rule_neural_act_print(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}
 
void rule_neural_act_rand(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}
  
void rule_neural_act_cover(XCSF *xcsf, CL *c, double *x, int action)
{
    do {
        rule_neural_cond_rand(xcsf, c);
    } while(!rule_neural_cond_match(xcsf, c, x) 
            && rule_neural_act_compute(xcsf, c, x) != action);
}
 
int rule_neural_act_compute(XCSF *xcsf, CL *c, double *x)
{
    (void)x; // network already updated
    RULE_NEURAL *cond = c->cond;
    c->action = 0;
    for(int i = 0; i < cond->num_outputs; i++) {
        if(neural_output(xcsf, &cond->net, i+1) > 0.5) {
            c->action += pow(2,i);
        }
    }
    c->action = iconstrain(0, xcsf->num_actions-1, c->action);
    return c->action;
}                

void rule_neural_act_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool rule_neural_act_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_neural_act_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_neural_act_mutate(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
    return false;
}

size_t rule_neural_act_save(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}

size_t rule_neural_act_load(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}
