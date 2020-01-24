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
 * @file rule_dgp.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Dynamical GP graph rule (condition + action) functions.
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
#include "action.h"
#include "rule_dgp.h"

/**
 * @brief Dynamical GP graph rule data structure.
 */ 
typedef struct RULE_DGP{
    GRAPH dgp; //!< DGP graph
    int num_outputs; //!< Number of action nodes (binarised)
} RULE_DGP;

/* CONDITION FUNCTIONS */

static void rule_dgp_cond_rand(XCSF *xcsf, CL *c);

void rule_dgp_cond_init(XCSF *xcsf, CL *c)
{
    RULE_DGP *new = malloc(sizeof(RULE_DGP));
    new->num_outputs = fmax(1, ceil(log2(xcsf->num_actions)));
    int n = fmax(xcsf->DGP_NUM_NODES, new->num_outputs + 1);
    graph_init(xcsf, &new->dgp, n);
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
    new->num_outputs = from_cond->num_outputs;
    to->cond = new;
}

static void rule_dgp_cond_rand(XCSF *xcsf, CL *c)
{
    RULE_DGP *cond = c->cond;
    graph_rand(xcsf, &cond->dgp);
}

void rule_dgp_cond_cover(XCSF *xcsf, CL *c, const double *x)
{
    (void)xcsf; (void)c; (void)x;
}

void rule_dgp_cond_update(XCSF *xcsf, CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool rule_dgp_cond_match(XCSF *xcsf, CL *c, const double *x)
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

size_t rule_dgp_cond_save(XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_DGP *cond = c->cond;
    size_t s = graph_save(xcsf, &cond->dgp, fp);
    return s;
}

size_t rule_dgp_cond_load(XCSF *xcsf, CL *c, FILE *fp)
{
    RULE_DGP *new = malloc(sizeof(RULE_DGP));
    size_t s = graph_load(xcsf, &new->dgp, fp);
    new->num_outputs = fmax(1, ceil(log2(xcsf->num_actions)));
    c->cond = new;
    return s;
}

/* ACTION FUNCTIONS */

void rule_dgp_act_init(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}

void rule_dgp_act_free(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}
 
void rule_dgp_act_copy(XCSF *xcsf, CL *to, CL *from)
{
    (void)xcsf; (void)to; (void)from;
}
 
void rule_dgp_act_print(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}
 
void rule_dgp_act_rand(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
}
  
void rule_dgp_act_cover(XCSF *xcsf, CL *c, const double *x, int action)
{
    do {
        rule_dgp_cond_rand(xcsf, c);
    } while(!rule_dgp_cond_match(xcsf, c, x) 
            && rule_dgp_act_compute(xcsf, c, x) != action);
}
 
int rule_dgp_act_compute(XCSF *xcsf, CL *c, const double *x)
{
    (void)x; // graph already updated
    RULE_DGP *cond = c->cond;
    c->action = 0;
    for(int i = 0; i < cond->num_outputs; i++) {
        if(graph_output(xcsf, &cond->dgp, i+1) > 0.5) {
            c->action += pow(2,i);
        }
    }
    c->action = iconstrain(0, xcsf->num_actions-1, c->action);
    return c->action;
}                

void rule_dgp_act_update(XCSF *xcsf, CL *c, const double *x, const double *y)
{
    (void)xcsf; (void)c; (void)x; (void)y;
}

_Bool rule_dgp_act_crossover(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_dgp_act_general(XCSF *xcsf, CL *c1, CL *c2)
{
    (void)xcsf; (void)c1; (void)c2;
    return false;
}

_Bool rule_dgp_act_mutate(XCSF *xcsf, CL *c)
{
    (void)xcsf; (void)c;
    return false;
}

size_t rule_dgp_act_save(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}

size_t rule_dgp_act_load(XCSF *xcsf, CL *c, FILE *fp)
{
    (void)xcsf; (void)c; (void)fp;
    return 0;
}
