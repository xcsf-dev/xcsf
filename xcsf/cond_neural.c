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
 * @file cond_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2019.
 * @brief Multi-layer perceptron neural network condition functions.
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
#include "condition.h"
#include "cond_neural.h"

/**
 * @brief Multi-layer perceptron neural network condition data structure.
 */ 
typedef struct COND_NEURAL {
    NET net; //!< Neural network
} COND_NEURAL;

void cond_neural_rand(XCSF *xcsf, CL *c);

void cond_neural_init(XCSF *xcsf, CL *c)
{
    COND_NEURAL *new = malloc(sizeof(COND_NEURAL));
    neural_init(xcsf, &new->net);

    // weights
    u_int32_t lopt = 0;
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
    l = neural_layer_connected_init(xcsf, hmax, 1, 1, LOGISTIC, lopt);
    neural_layer_insert(xcsf, &new->net, l, 1); 

    c->cond = new;
}

void cond_neural_free(XCSF *xcsf, CL *c)
{
    COND_NEURAL *cond = c->cond;
    neural_free(xcsf, &cond->net);
    free(c->cond);
}                  

void cond_neural_copy(XCSF *xcsf, CL *to, CL *from)
{
    COND_NEURAL *new = malloc(sizeof(COND_NEURAL));
    COND_NEURAL *from_cond = from->cond;
    neural_copy(xcsf, &new->net, &from_cond->net);
    to->cond = new;
}

void cond_neural_rand(XCSF *xcsf, CL *c)
{
    COND_NEURAL *cond = c->cond;
    neural_rand(xcsf, &cond->net);
}

void cond_neural_cover(XCSF *xcsf, CL *c, double *x)
{
    do {
        cond_neural_rand(xcsf, c);
    } while(!cond_neural_match(xcsf, c, x));
}

void cond_neural_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool cond_neural_match(XCSF *xcsf, CL *c, double *x)
{
    COND_NEURAL *cond = c->cond;
    neural_propagate(xcsf, &cond->net, x);
    if(neural_output(xcsf, &cond->net, 0) > 0.5) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}                

_Bool cond_neural_mutate(XCSF *xcsf, CL *c)
{
    COND_NEURAL *cond = c->cond;
    return neural_mutate(xcsf, &cond->net);
}

_Bool cond_neural_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void) c1; (void)c2;
    return false;
}

_Bool cond_neural_general(XCSF *xcsf, CL *c1, CL *c2)
{
    if(c1->exp < xcsf->THETA_SUB || c2->exp < xcsf->THETA_SUB) {
        return false;
    }
    for(int i = 0; i < xcsf->THETA_SUB; i++) {
        int i1 = (c1->exp + i) % xcsf->THETA_SUB;
        int i2 = (c2->exp + i) % xcsf->THETA_SUB;
        if(c1->mhist[i1] == false && c2->mhist[i2] == true) {
            return false;
        }
    }
    return true;
}   

void cond_neural_print(XCSF *xcsf, CL *c)
{
    COND_NEURAL *cond = c->cond;
    neural_print(xcsf, &cond->net, false);
}

int cond_neural_size(XCSF *xcsf, CL *c)
{
    COND_NEURAL *cond = c->cond;
    return neural_size(xcsf, &cond->net);
}

size_t cond_neural_save(XCSF *xcsf, CL *c, FILE *fp)
{
    COND_NEURAL *cond = c->cond;
    size_t s = neural_save(xcsf, &cond->net, fp);
    //printf("cond neural saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t cond_neural_load(XCSF *xcsf, CL *c, FILE *fp)
{
    COND_NEURAL *new = malloc(sizeof(COND_NEURAL));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->cond = new;
    //printf("cond neural loaded %lu elements\n", (unsigned long)s);
    return s;
}
