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
 * @file pred_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2019.
 * @brief Multi-layer perceptron neural network prediction functions.
 */ 
 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_noise.h"
#include "neural_layer_softmax.h"
#include "prediction.h"
#include "pred_neural.h"
                                          
/**
 * @brief Multi-layer perceptron neural network prediction data structure.
 */ 
typedef struct PRED_NEURAL {
    NET net; //!< Neural network
} PRED_NEURAL;

void pred_neural_init(XCSF *xcsf, CL *c)
{
    PRED_NEURAL *new = malloc(sizeof(PRED_NEURAL));
    neural_init(xcsf, &new->net);
 
    // weights
    uint32_t lopt = 0;
    if(xcsf->PRED_EVOLVE_WEIGHTS) {
        lopt |= LAYER_EVOLVE_WEIGHTS;
    }
    if(xcsf->PRED_EVOLVE_ETA) {
        lopt |= LAYER_EVOLVE_ETA;
    }
    if(xcsf->PRED_SGD_WEIGHTS) {
        lopt |= LAYER_SGD_WEIGHTS;
    }
    // neurons
    int hmax = fmax(xcsf->PRED_MAX_HIDDEN_NEURONS, 1);
    int hinit = xcsf->PRED_NUM_HIDDEN_NEURONS;
    if(hinit < 1) {
        hinit = irand_uniform(1, hmax);
    }
    if(hmax < hinit) {
        hmax = hinit;
    }
    if(xcsf->PRED_EVOLVE_NEURONS) {
        lopt |= LAYER_EVOLVE_NEURONS;
    }
    else {
        hmax = hinit;
    }
    // functions
    if(xcsf->PRED_EVOLVE_FUNCTIONS) {
        lopt |= LAYER_EVOLVE_FUNCTIONS;
    }

    // hidden layer
    int f = xcsf->PRED_HIDDEN_NEURON_ACTIVATION;
    LAYER *l = neural_layer_connected_init(xcsf, xcsf->num_x_vars, hinit, hmax, f, lopt);
    neural_layer_insert(xcsf, &new->net, l, 0); 

    // output layer
    lopt &= ~LAYER_EVOLVE_NEURONS; // never evolve the number of output neurons
    l = neural_layer_connected_init(xcsf, hmax, xcsf->num_y_vars, xcsf->num_y_vars, LOGISTIC, lopt);
    neural_layer_insert(xcsf, &new->net, l, 1); 

    c->pred = new;

    //l = neural_layer_noise_init(xcsf, xcsf->num_x_vars, 0.5, 0.5);
    //l = neural_layer_dropout_init(xcsf, xcsf->num_x_vars, 0.2);
    //neural_layer_insert(xcsf, &new->net, l, 0); 
    //l = neural_layer_softmax_init(xcsf, xcsf->num_y_vars, 1);
    //neural_layer_insert(xcsf, &new->net, l, 4);
}

void pred_neural_free(XCSF *xcsf, CL *c)
{
    PRED_NEURAL *pred = c->pred;
    neural_free(xcsf, &pred->net);
    free(pred);
}

void pred_neural_copy(XCSF *xcsf, CL *to, CL *from)
{
    PRED_NEURAL *new = malloc(sizeof(PRED_NEURAL));
    PRED_NEURAL *from_pred = from->pred;
    neural_copy(xcsf, &new->net, &from_pred->net);
    to->pred = new;
}

void pred_neural_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    if(xcsf->PRED_SGD_WEIGHTS) {
        PRED_NEURAL *pred = c->pred;
        neural_learn(xcsf, &pred->net, y, x);
    }
}

double *pred_neural_compute(XCSF *xcsf, CL *c, double *x)
{
    PRED_NEURAL *pred = c->pred;
    neural_propagate(xcsf, &pred->net, x);
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        c->prediction[i] = neural_output(xcsf, &pred->net, i);
    }
    return c->prediction;
}

void pred_neural_print(XCSF *xcsf, CL *c)
{
    PRED_NEURAL *pred = c->pred;
    neural_print(xcsf, &pred->net, false);
}  

_Bool pred_neural_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void) c1; (void)c2;
    return false;
}

_Bool pred_neural_mutate(XCSF *xcsf, CL *c)
{
    PRED_NEURAL *pred = c->pred;
    return neural_mutate(xcsf, &pred->net);
}

int pred_neural_size(XCSF *xcsf, CL *c)
{
    PRED_NEURAL *pred = c->pred;
    return neural_size(xcsf, &pred->net);
}

size_t pred_neural_save(XCSF *xcsf, CL *c, FILE *fp)
{
    PRED_NEURAL *pred = c->pred;
    size_t s = neural_save(xcsf, &pred->net, fp);
    //printf("pred neural saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t pred_neural_load(XCSF *xcsf, CL *c, FILE *fp)
{
    PRED_NEURAL *new = malloc(sizeof(PRED_NEURAL));
    size_t s = neural_load(xcsf, &new->net, fp);
    c->pred = new;
    //printf("pred neural loaded %lu elements\n", (unsigned long)s);
    return s;
}

double pred_neural_eta(XCSF *xcsf, CL *c, int layer)
{
    (void)xcsf;
    PRED_NEURAL *pred = c->pred;
    NET *net = &pred->net;
    int i = 0;
    for(LLIST *iter = net->tail; iter != NULL; iter = iter->prev) {
        if(i == layer) {
            return iter->layer->eta;
        }
        i++;
    }
    return 0;
}
