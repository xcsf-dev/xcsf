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
 * The DGP classifier rule module.
 * Performs both condition matching and prediction in a single evolved graph.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "dgp.h"
#include "condition.h"
#include "prediction.h"
#include "rule_dgp.h"

typedef struct RULE_DGP{
    GRAPH dgp;
} RULE_DGP;

void rule_dgp_cond_rand(XCSF *xcsf, CL *c);

void rule_dgp_cond_init(XCSF *xcsf, CL *c)
{
    RULE_DGP *new = malloc(sizeof(RULE_DGP));
    graph_init(xcsf, &new->dgp, xcsf->DGP_NUM_NODES);
    graph_rand(xcsf, &new->dgp);
    c->cond = new;
}

void rule_dgp_cond_free(XCSF *xcsf, CL *c)
{
    RULE_DGP *cond = c->cond;
    graph_free(xcsf, &cond->dgp);
    free(c->cond);
}

void rule_dgp_cond_copy(XCSF *xcsf, CL *to, CL *from)
{
    RULE_DGP *new = malloc(sizeof(RULE_DGP));
    RULE_DGP *from_cond = from->cond;
    graph_init(xcsf, &new->dgp, from_cond->dgp.n);
    graph_copy(xcsf, &new->dgp, &from_cond->dgp);
    to->cond = new;
}

void rule_dgp_cond_rand(XCSF *xcsf, CL *c)
{
    RULE_DGP *cond = c->cond;
    graph_rand(xcsf, &cond->dgp);
}

void rule_dgp_cond_cover(XCSF *xcsf, CL *c, double *x)
{
    do {
        rule_dgp_cond_rand(xcsf, c);
    } while(!rule_dgp_cond_match(xcsf, c, x));
}

void rule_dgp_cond_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool rule_dgp_cond_match(XCSF *xcsf, CL *c, double *x)
{
    RULE_DGP *cond = c->cond;
    graph_update(xcsf, &cond->dgp, x);
    if(graph_output(xcsf, &cond->dgp, 0) > 0.5) {
        c->m = true;
    }
    else {
        c->m = false;
    }
    return c->m;
}    

_Bool rule_dgp_cond_mutate(XCSF *xcsf, CL *c)
{
    RULE_DGP *cond = c->cond;
    return graph_mutate(xcsf, &cond->dgp);
}

_Bool rule_dgp_cond_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    RULE_DGP *cond1 = c1->cond;
    RULE_DGP *cond2 = c2->cond;
    return graph_crossover(xcsf, &cond1->dgp, &cond2->dgp);
}

_Bool rule_dgp_cond_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}   

void rule_dgp_cond_print(XCSF *xcsf, CL *c)
{
    RULE_DGP *cond = c->cond;
    graph_print(xcsf, &cond->dgp);
}  
 
int rule_dgp_cond_size(XCSF *xcsf, CL *c)
{
    (void)xcsf;
    RULE_DGP *cond = c->cond;
    return cond->dgp.n;
}

void rule_dgp_pred_init(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void rule_dgp_pred_free(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void rule_dgp_pred_copy(XCSF *xcsf, CL *to, CL *from)
{
    (void)xcsf; (void)to; (void)from;
}

void rule_dgp_pred_update(XCSF *xcsf, CL *c, double *x, double *y)
{
    (void)xcsf; (void)c; (void)y; (void)x;
}

double *rule_dgp_pred_compute(XCSF *xcsf, CL *c, double *x)
{
    (void)x;
    RULE_DGP *cond = c->cond;
    for(int i = 0; i < xcsf->num_y_vars; i++) {
        c->prediction[i] = graph_output(xcsf, &cond->dgp, 1+i);
    }
    return c->prediction;
}

void rule_dgp_pred_print(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

_Bool rule_dgp_pred_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_dgp_pred_mutate(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
    return false;
}

int rule_dgp_pred_size(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
    return 0;
}

size_t rule_dgp_cond_save(XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_DGP *cond = c->cond;
    size_t s = graph_save(xcsf, &cond->dgp, fp);
    //printf("rule dgp saved %lu elements\n", (unsigned long)s);
    return s;
}

size_t rule_dgp_cond_load(XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_DGP *cond = c->cond;
    size_t s = graph_load(xcsf, &cond->dgp, fp);
    //printf("rule dgp loaded %lu elements\n", (unsigned long)s);
    return s;
}

size_t rule_dgp_pred_save(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}

size_t rule_dgp_pred_load(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}
