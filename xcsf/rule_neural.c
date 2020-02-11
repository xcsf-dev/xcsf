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
 * @date 2019--2020.
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
    int n_outputs; //!< Number of action nodes (binarised)
} RULE_NEURAL;

/* CONDITION FUNCTIONS */

static void rule_neural_cond_rand(const XCSF *xcsf, const CL *c);
static uint32_t rule_neural_lopt(const XCSF *xcsf);

void rule_neural_cond_init(const XCSF *xcsf, CL *c)
{
    RULE_NEURAL *new = malloc(sizeof(RULE_NEURAL));
    neural_init(xcsf, &new->net);
    // hidden layers
    uint32_t lopt = rule_neural_lopt(xcsf);
    LAYER *l;
    int i = 0;
    int n_inputs = xcsf->x_dim;
    while(i < MAX_LAYERS && xcsf->COND_NUM_NEURONS[i] > 0) {
        int hinit = xcsf->COND_NUM_NEURONS[i];
        int hmax = xcsf->COND_MAX_NEURONS[i];
        if(hmax < hinit || !xcsf->COND_EVOLVE_NEURONS) {
            hmax = hinit;
        }
        int f = xcsf->COND_HIDDEN_ACTIVATION;
        l = neural_layer_connected_init(xcsf, n_inputs, hinit, hmax, f, lopt);
        neural_layer_insert(xcsf, &new->net, l, i);
        n_inputs = hinit;
        i++;
    }
    // output layer
    int f = xcsf->COND_OUTPUT_ACTIVATION;
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    int n = fmax(1, ceil(log2(xcsf->n_actions))); // number of action neurons
    new->n_outputs = n;
    l = neural_layer_connected_init(xcsf, n_inputs, n+1, n+1, f, lopt);
    neural_layer_insert(xcsf, &new->net, l, i);
    c->cond = new;
}

static uint32_t rule_neural_lopt(const XCSF *xcsf)
{
    uint32_t lopt = 0;
    if(xcsf->COND_EVOLVE_WEIGHTS) {
        lopt |= LAYER_EVOLVE_WEIGHTS;
    }
    if(xcsf->COND_EVOLVE_NEURONS) {
        lopt |= LAYER_EVOLVE_NEURONS;
    }
    if(xcsf->COND_EVOLVE_FUNCTIONS) {
        lopt |= LAYER_EVOLVE_FUNCTIONS;
    }
    return lopt;
}

void rule_neural_cond_free(const XCSF *xcsf, const CL *c)
{
    RULE_NEURAL *cond = c->cond;
    neural_free(xcsf, &cond->net);
    free(c->cond);
}

void rule_neural_cond_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    RULE_NEURAL *new = malloc(sizeof(RULE_NEURAL));
    const RULE_NEURAL *from_cond = from->cond;
    new->n_outputs = from_cond->n_outputs;
    neural_copy(xcsf, &new->net, &from_cond->net);
    to->cond = new;
}

static void rule_neural_cond_rand(const XCSF *xcsf, const CL *c)
{
    const RULE_NEURAL *cond = c->cond;
    neural_rand(xcsf, &cond->net);
}

void rule_neural_cond_cover(const XCSF *xcsf, const CL *c, const double *x)
{
    (void)xcsf; (void)c; (void)x;
}

void rule_neural_cond_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool rule_neural_cond_match(const XCSF *xcsf, const CL *c, const double *x)
{
    const RULE_NEURAL *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if(neural_output(xcsf, &cond->net, 0) > 0.5) {
        return true;
    }
    else {
        return false;
    }
}    

_Bool rule_neural_cond_mutate(const XCSF *xcsf, const CL *c)
{
    const RULE_NEURAL *cond = c->cond;
    return neural_mutate(xcsf, &cond->net);
}

_Bool rule_neural_cond_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void) c1; (void)c2;
    return false;
}

_Bool rule_neural_cond_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}   

void rule_neural_cond_print(const XCSF *xcsf, const CL *c)
{
    const RULE_NEURAL *cond = c->cond;
    neural_print(xcsf, &cond->net, false);
}  

int rule_neural_cond_size(const XCSF *xcsf, const CL *c)
{
    const RULE_NEURAL *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t rule_neural_cond_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    const RULE_NEURAL *cond = c->cond;
    size_t s = neural_save(xcsf, &cond->net, fp);
    return s;
}

size_t rule_neural_cond_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_NEURAL *new = malloc(sizeof(RULE_NEURAL));
    size_t s = neural_load(xcsf, &new->net, fp);
    new->n_outputs = fmax(1, ceil(log2(xcsf->n_actions)));
    c->cond = new;
    return s;
}

/* ACTION FUNCTIONS */

void rule_neural_act_init(const XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void rule_neural_act_free(const XCSF *xcsf, const CL *c)
{
    (void)xcsf; (void)c;
}

void rule_neural_act_copy(const XCSF *xcsf, CL *to, const CL *from)
{
    (void)xcsf; (void)to; (void)from;
}

void rule_neural_act_print(const XCSF *xcsf, const CL *c)
{
    (void)xcsf; (void)c;
}

void rule_neural_act_rand(const XCSF *xcsf, const CL *c)
{
    (void)xcsf; (void)c;
}

void rule_neural_act_cover(const XCSF *xcsf, const CL *c, const double *x, int action)
{
    do {
        rule_neural_cond_rand(xcsf, c);
    } while(!rule_neural_cond_match(xcsf, c, x) 
            && rule_neural_act_compute(xcsf, c, x) != action);
}

int rule_neural_act_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    (void)x; // network already updated
    const RULE_NEURAL *cond = c->cond;
    int action = 0;
    for(int i = 0; i < cond->n_outputs; i++) {
        if(neural_output(xcsf, &cond->net, i+1) > 0.5) {
            action += pow(2,i);
        }
    }
    action = iconstrain(0, xcsf->n_actions-1, action);
    return action;
}                

void rule_neural_act_update(const XCSF *xcsf, const CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool rule_neural_act_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_neural_act_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_neural_act_mutate(const XCSF *xcsf, const CL *c)
{
    (void)xcsf; (void)c;
    return false;
}

size_t rule_neural_act_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}

size_t rule_neural_act_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}
