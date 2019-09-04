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

typedef struct PRED_NEURAL {
    NET net;
} PRED_NEURAL;

void pred_neural_init(XCSF *xcsf, CL *c)
{
    PRED_NEURAL *new = malloc(sizeof(PRED_NEURAL));
    neural_init(xcsf, &new->net);
    LAYER *l;

    //l = neural_layer_noise_init(xcsf, xcsf->num_x_vars, 0.5, 0.5);
    //l = neural_layer_dropout_init(xcsf, xcsf->num_x_vars, 0.2);
    //neural_layer_insert(xcsf, &new->net, l, 0); 

    l = neural_layer_connected_init(xcsf,
            xcsf->num_x_vars, xcsf->NUM_HIDDEN_NEURONS, xcsf->HIDDEN_NEURON_ACTIVATION, 1);
    neural_layer_insert(xcsf, &new->net, l, 0); 

    //l = neural_layer_dropout_init(xcsf, xcsf->NUM_HIDDEN_NEURONS, 0.5);
    //neural_layer_insert(xcsf, &new->net, l, 2); 

    l = neural_layer_connected_init(xcsf, xcsf->NUM_HIDDEN_NEURONS, xcsf->num_y_vars, LOGISTIC,0);
    neural_layer_insert(xcsf, &new->net, l, 1); 

    //l = neural_layer_softmax_init(xcsf, xcsf->num_y_vars, 1);
    //neural_layer_insert(xcsf, &new->net, l, 4);

    c->pred = new;
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
    PRED_NEURAL *pred = c->pred;
    neural_learn(xcsf, &pred->net, y, x);
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
    PRED_NEURAL *pred1 = c1->pred;
    PRED_NEURAL *pred2 = c2->pred;
    return neural_crossover(xcsf, &pred1->net, &pred2->net);
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
