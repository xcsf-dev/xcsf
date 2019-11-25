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
 * @file pa.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2019.
 * @brief Prediction array functions.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "cl_set.h"
#include "pa.h"

void pa_init(XCSF *xcsf)
{
    xcsf->pa = malloc(sizeof(double) * xcsf->num_actions);
    xcsf->nr = malloc(sizeof(double) * xcsf->num_actions);
}

void pa_build(XCSF *xcsf, SET *set, double *x)
{
    for(int i = 0; i < xcsf->num_actions; i++) {
        xcsf->pa[i] = 0;
        xcsf->nr[i] = 0;
    }
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        double *predictions = cl_predict(xcsf, iter->cl, x);
        xcsf->pa[iter->cl->action] += predictions[0] * iter->cl->fit;
        xcsf->nr[iter->cl->action] += iter->cl->fit;
    }
    for(int i = 0; i < xcsf->num_actions; i++) {
        if(xcsf->nr[i] != 0) {
            xcsf->pa[i] /= xcsf->nr[i];
        }
        else {
            xcsf->pa[i] = 0;
        }
    }
}

int pa_best_action(XCSF *xcsf)
{
    int action = 0;
    for(int i = 1; i < xcsf->num_actions; i++) {
        if(xcsf->pa[action] < xcsf->pa[i]) {
            action = i;
        }
    }
    return action;
}

int pa_rand_action(XCSF *xcsf)
{
    int action = 0;
    do {
        action = irand_uniform(0, xcsf->num_actions);
    } while(xcsf->nr[action] == 0);
    return action;
}

double pa_best_val(XCSF *xcsf)
{
    double max = xcsf->pa[0];
    for(int i = 1; i < xcsf->num_actions; i++) {
        if(max < xcsf->pa[i]) {
            max = xcsf->pa[i];
        }
    }
    return max;
}

double pa_val(XCSF *xcsf, int action)
{
    if(action >= 0 && action < xcsf->num_actions) {
        return xcsf->pa[action];
    }
    return -1;
}

void pa_free(XCSF *xcsf)
{
    free(xcsf->pa);
    free(xcsf->nr);
}
