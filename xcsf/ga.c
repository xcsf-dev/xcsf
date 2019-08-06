/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
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
 **************
 * Description: 
 **************
 * The genetic algorithm module.
 *
 * Selects parents to create offspring via crossover and mutation, and inserts
 * the newly created classifiers into the population. The maximum population
 * size limit is then enforced by deleting excess classifiers from the
 * population. Performs GA subsumption if enabled.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "data_structures.h"
#include "random.h"
#include "cl.h"
#include "cl_set.h"    
#include "ga.h"

CL *ga_select_parent(XCSF *xcsf, NODE **set, double fit_sum);
void ga_subsume(XCSF *xcsf, CL *c, CL *c1p, CL *c2p, NODE **set, int size);

void ga(XCSF *xcsf, NODE **set, int size, int num, int time, NODE **kset)
{
    // check if the genetic algorithm should be run
    if(size == 0 || time - set_mean_time(xcsf, set, num) < xcsf->THETA_GA) {
        return;
    }
    set_times(xcsf, set, time);
    // select parents
    double fit_sum = set_total_fit(xcsf, set);
    CL *c1p = ga_select_parent(xcsf, set, fit_sum);
    CL *c2p = ga_select_parent(xcsf, set, fit_sum);

    for(int i = 0; i < xcsf->THETA_OFFSPRING/2; i++) {
        // create copies of parents
        CL *c1 = malloc(sizeof(CL));
        CL *c2 = malloc(sizeof(CL));
        cl_copy(xcsf, c1, c1p);
        cl_copy(xcsf, c2, c2p);
        // reduce offspring err, fit
        c1->err = xcsf->ERR_REDUC * ((c1p->err + c2p->err)/2.0);
        c2->err = c1->err;
        c1->fit = c1p->fit / c1p->num;
        c2->fit = c2p->fit / c2p->num;
        c1->fit = xcsf->FIT_REDUC * (c1->fit + c2->fit)/2.0;
        c2->fit = c1->fit;

        switch(xcsf->COND_TYPE) {
            // conditions that do not fully support subsumption or crossover
            case -1:
            case 1:
            case 2:
            case 3:
            case 11:
            case 12:
                if(!cl_mutate(xcsf, c1) && xcsf->GA_SUBSUMPTION) {
                    c1p->num++;
                    xcsf->pop_num_sum++;
                    cl_free(xcsf, c1);
                }
                else {
                    pop_add(xcsf, c1);
                }
                if(!cl_mutate(xcsf, c2) && xcsf->GA_SUBSUMPTION) {
                    c2p->num++;
                    xcsf->pop_num_sum++;
                    cl_free(xcsf, c2);
                }
                else {
                    pop_add(xcsf, c2);
                }
                break;

            default:
                // apply genetic operators to offspring
                cl_crossover(xcsf, c1, c2);
                cl_mutate(xcsf, c1);
                cl_mutate(xcsf, c2);
                // add offspring to population
                if(xcsf->GA_SUBSUMPTION) {
                    ga_subsume(xcsf, c1, c1p, c2p, set, size);
                    ga_subsume(xcsf, c2, c1p, c2p, set, size);
                }
                else {
                    pop_add(xcsf, c1);
                    pop_add(xcsf, c2);
                }
                break;
        }
    }
    pop_enforce_limit(xcsf, kset);
}   

void ga_subsume(XCSF *xcsf, CL *c, CL *c1p, CL *c2p, NODE **set, int size)
{
    // check if either parent subsumes the offspring
    if(cl_subsumer(xcsf, c1p) && cl_subsumes(xcsf, c1p, c)) {
        c1p->num++;
        xcsf->pop_num_sum++;
        cl_free(xcsf, c);
    }
    else if(cl_subsumer(xcsf, c2p) && cl_subsumes(xcsf, c2p, c)) {
        c2p->num++;
        xcsf->pop_num_sum++;
        cl_free(xcsf, c);
    }
    // attempt to find a random subsumer from the set
    else {
        NODE *candidates[size];
        int choices = 0;
        for(NODE *iter = *set; iter != NULL; iter = iter->next) {
            if(cl_subsumer(xcsf, iter->cl) 
                    && cl_subsumes(xcsf, iter->cl, c)) {
                candidates[choices] = iter;
                choices++;
            }
        }
        // found
        if(choices > 0) {
            candidates[irand(0,choices)]->cl->num++;
            xcsf->pop_num_sum++;
            cl_free(xcsf, c);
        }
        // if no subsumers are found the offspring is added to the population
        else {
            pop_add(xcsf, c);   
        }
    }
}

CL *ga_select_parent(XCSF *xcsf, NODE **set, double fit_sum)
{
    (void)xcsf;
    // selects a classifier using roullete wheel selection with the fitness
    // (a fitness proportionate selection mechanism.)
    double p = drand() * fit_sum;
    NODE *iter = *set;
    double sum = iter->cl->fit;
    while(p > sum) {
        iter = iter->next;
        sum += iter->cl->fit;
    }
    return iter->cl;
}
