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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "cl_set.h"

void set_subsumption(XCSF *xcsf, SET *set, SET *kset);
void set_update_fit(XCSF *xcsf, SET *set);

void pop_init(XCSF *xcsf)
{
    xcsf->time = 0; // number of learning trials performed
    xcsf->msetsize = 0.0; // average match set size
    set_init(xcsf, &xcsf->pset);
    // initialise population
    if(xcsf->POP_INIT) {
        while(xcsf->pset.num < xcsf->POP_SIZE) {
            CL *new = malloc(sizeof(CL));
            cl_init(xcsf, new, xcsf->POP_SIZE, 0);
            cl_rand(xcsf, new);
            set_add(xcsf, &xcsf->pset, new);
        }
    }
}

void set_init(XCSF *xcsf, SET *set)
{
    (void)xcsf;
    set->list = NULL;
    set->size = 0;
    set->num = 0;
}

void pop_del(XCSF *xcsf, SET *kset)
{
    // selects a classifier using roullete wheel selection with the deletion 
    // vote; sets its numerosity to zero, and removes it from the population 

    // select a roullete point
    double avg_fit = set_total_fit(xcsf, &xcsf->pset) / xcsf->pset.num;
    double sum = 0.0;
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        sum += cl_del_vote(xcsf, iter->cl, avg_fit);
    }
    double p = rand_uniform(0,sum);

    // find the classifier to delete using the point
    sum = 0.0;
    CLIST *prev = NULL;
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        sum += cl_del_vote(xcsf, iter->cl, avg_fit);
        if(sum > p) {
            (iter->cl->num)--;
            (xcsf->pset.num)--;
            // macro classifier must be deleted
            if(iter->cl->num == 0) {
                set_add(xcsf, kset, iter->cl);
                (xcsf->pset.size)--;
                if(prev == NULL) {
                    xcsf->pset.list = iter->next;
                }
                else {
                    prev->next = iter->next;    
                }
                free(iter);
            }
            return;
        }
        prev = iter; 
    }   
}

void pop_enforce_limit(XCSF *xcsf, SET *kset)
{
    while(xcsf->pset.num > xcsf->POP_SIZE) {
        pop_del(xcsf, kset);
    }
}

void set_match(XCSF *xcsf, SET *mset, SET *kset, double *x)
{
    // add classifiers that match the input state to the match set  
#ifdef PARALLEL_MATCH
    CLIST *blist[xcsf->pset.size];
    int j = 0;
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
    // update current matching conditions setting m flags
    #pragma omp parallel for
    for(int i = 0; i < xcsf->pset.size; i++) {
        cl_match(xcsf, blist[i]->cl, x);
    }
    // build m list
    for(int i = 0; i < xcsf->pset.size; i++) {
        if(cl_m(xcsf, blist[i]->cl)) {
            set_add(xcsf, mset, blist[i]->cl);
        }
    }
#else
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        if(cl_match(xcsf, iter->cl, x)) {
            set_add(xcsf, mset, iter->cl);
        }
    }   
#endif
    // perform covering if match set size is < THETA_MNA
    while(mset->size < xcsf->THETA_MNA) {
        // new classifier with matching condition
        CL *new = malloc(sizeof(CL));
        cl_init(xcsf, new, (mset->num)+1, xcsf->time);
        cl_cover(xcsf, new, x);
        set_add(xcsf, &xcsf->pset, new);
        set_add(xcsf, mset, new); 
        pop_enforce_limit(xcsf, kset);
        // remove any deleted classifiers from the match set
        set_validate(xcsf, mset);
    }
}

void set_pred(XCSF *xcsf, SET *set, double *x, double *p)
{
    // match set fitness weighted prediction
    double *presum = calloc(xcsf->num_y_vars, sizeof(double));
    double fitsum = 0.0;
#ifdef PARALLEL_PRED
    CLIST *blist[set->size];
    int j = 0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
    #pragma omp parallel for reduction(+:presum[:xcsf->num_y_vars],fitsum)
    for(int i = 0; i < set->size; i++) {
        double *predictions = cl_predict(xcsf, blist[i]->cl, x);
        for(int var = 0; var < xcsf->num_y_vars; var++) {
            presum[var] += predictions[var] * blist[i]->cl->fit;
        }
        fitsum += blist[i]->cl->fit;
    }
    #pragma omp parallel for
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        p[var] = presum[var]/fitsum;
    }
#else
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        double *predictions = cl_predict(xcsf, iter->cl, x);
        for(int var = 0; var < xcsf->num_y_vars; var++) {
            presum[var] += predictions[var] * iter->cl->fit;
        }
        fitsum += iter->cl->fit;
    }    
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        p[var] = presum[var]/fitsum;
    }
#endif
    // clean up
    free(presum);
}

void set_add(XCSF *xcsf, SET *set, CL *c)
{
    (void)xcsf;
    // adds a classifier to the set
    if(set->list == NULL) {
        set->list = malloc(sizeof(CLIST));
        set->list->cl = c;
        set->list->next = NULL;
    }
    else {
        CLIST *new = malloc(sizeof(CLIST));
        new->cl = c;
        new->next = set->list;
        set->list = new;
    }
    set->size++;
    set->num++;
}

void set_update(XCSF *xcsf, SET *set, SET *kset, double *x, double *y)
{
#ifdef PARALLEL_UPDATE
    CLIST *blist[set->size];
    int j = 0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
    #pragma omp parallel for
    for(int i = 0; i < set->size; i++) {
        cl_update(xcsf, blist[i]->cl, x, y, set->num);
    }
#else
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        cl_update(xcsf, iter->cl, x, y, set->num);
    }
#endif
    set_update_fit(xcsf, set);
    if(xcsf->SET_SUBSUMPTION) {
        set_subsumption(xcsf, set, kset);
    }
}

void set_update_fit(XCSF *xcsf, SET *set)
{
    double acc_sum = 0.0;
    double accs[set->size];
    // calculate accuracies
    int i = 0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        accs[i] = cl_acc(xcsf, iter->cl);
        acc_sum += accs[i] * set->num;
        i++;
    }
    // update fitnesses
    i = 0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        cl_update_fit(xcsf, iter->cl, acc_sum, accs[i]);
        i++;
    }
}

void set_subsumption(XCSF *xcsf, SET *set, SET *kset)
{
    CL *s = NULL;
    CLIST *iter;
    // find the most general subsumer in the set
    for(iter = set->list; iter != NULL; iter = iter->next) {
        CL *c = iter->cl;
        if(cl_subsumer(xcsf, c)) {
            if(s == NULL || cl_general(xcsf, c, s)) {
                s = c;
            }
        }
    }
    // subsume the more specific classifiers in the set
    if(s != NULL) {
        iter = set->list; 
        while(iter != NULL) {
            CL *c = iter->cl;
            iter = iter->next;
            if(s != c && cl_general(xcsf, s, c)) {
                s->num += c->num;
                c->num = 0;
                set_add(xcsf, kset, c);
                set_validate(xcsf, set);
                set_validate(xcsf, &xcsf->pset);
            }
        }
    }
}

void set_validate(XCSF *xcsf, SET *set)
{
    (void)xcsf;
    // remove nodes pointing to classifiers with 0 numerosity
    set->size = 0;
    set->num = 0;
    CLIST *prev = NULL;
    CLIST *iter = set->list;
    while(iter != NULL) {
        if(iter->cl == NULL || iter->cl->num == 0) {
            if(prev == NULL) {
                set->list = iter->next;
                free(iter);
                iter = set->list;
            }
            else {
                prev->next = iter->next;
                free(iter);
                iter = prev->next;
            }
        }
        else {
            set->size++;
            set->num += iter->cl->num;
            prev = iter;
            iter = iter->next;
        }
    }
}

void set_print(XCSF *xcsf, SET *set, _Bool print_cond, _Bool print_pred)
{
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        cl_print(xcsf, iter->cl, print_cond, print_pred);
    }
}

void set_times(XCSF *xcsf, SET *set)
{
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        iter->cl->time = xcsf->time;
    }
}

double set_total_fit(XCSF *xcsf, SET *set)
{
    (void)xcsf;
    double sum = 0.0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += iter->cl->fit;
    }
    return sum;
}

double set_total_time(XCSF *xcsf, SET *set)
{
    (void)xcsf;
    double sum = 0.0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += iter->cl->time * iter->cl->num;
    }
    return sum;
}

double set_mean_time(XCSF *xcsf, SET *set)
{
    return set_total_time(xcsf, set) / set->num;
}

void set_free(XCSF *xcsf, SET *set)
{
    (void)xcsf;
    // frees the set only, not the classifiers
    CLIST *iter = set->list;
    while(iter != NULL) {
        set->list = iter->next;
        free(iter);
        iter = set->list;
    }
}

void set_kill(XCSF *xcsf, SET *set)
{
    // frees the set and classifiers
    CLIST *iter = set->list;
    while(iter != NULL) {
        cl_free(xcsf, iter->cl);
        set->list = iter->next;
        free(iter);
        iter = set->list;
    }
}

double set_avg_mut(XCSF *xcsf, SET *set, int m)
{
    // return the fixed value if not adapted
    if(m >= xcsf->SAM_NUM) {
        switch(m) {
            case 0:
                return xcsf->S_MUTATION;
            case 1:
                return xcsf->P_MUTATION;
            case 2:
                return xcsf->P_FUNC_MUTATION;
            default:
                return -1;
        }
    }

    // returns the average classifier mutation rate
    double sum = 0.0;
    int cnt = 0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += cl_mutation_rate(xcsf, iter->cl, m);
        cnt++;
    }
    return sum/cnt;
}
