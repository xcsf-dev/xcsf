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
 * The classifier set module.
 *
 * Performs operations applied to sets of classifiers: creation, deletion,
 * updating, prediction, validation, printing.  
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "data_structures.h"
#include "random.h"
#include "cl.h"
#include "cl_set.h"

void set_subsumption(XCSF *xcsf, NODE **set, int *size, int *num, NODE **kset);
void set_update_fit(XCSF *xcsf, NODE **set, int size, int num_sum);

void pop_init(XCSF *xcsf)
{
    // initialise population
    xcsf->pset = NULL; // population linked list
    xcsf->pop_num = 0; // num macro-classifiers
    xcsf->pop_num_sum = 0; // numerosity sum

    if(xcsf->POP_INIT) {
        while(xcsf->pop_num_sum < xcsf->POP_SIZE) {
            CL *new = malloc(sizeof(CL));
            cl_init(xcsf, new, xcsf->POP_SIZE, 0);
            cl_rand(xcsf, new);
            pop_add(xcsf, new);
        }
    }
}

void pop_add(XCSF *xcsf, CL *c)
{
    // adds a classifier to the population set
    xcsf->pop_num_sum++;
    xcsf->pop_num++;
    if(xcsf->pset == NULL) {
        xcsf->pset = malloc(sizeof(NODE));
        xcsf->pset->cl = c;
        xcsf->pset->next = NULL;
    }
    else {
        NODE *new = malloc(sizeof(NODE));
        new->next = xcsf->pset;
        new->cl = c;
        xcsf->pset = new;
    }
}

void pop_del(XCSF *xcsf, NODE **kset)
{
    // selects a classifier using roullete wheel selection with the deletion 
    // vote; sets its numerosity to zero, and removes it from the population 

    // select a roullete point
    double avg_fit = set_total_fit(xcsf, &xcsf->pset) / xcsf->pop_num_sum;
    double sum = 0.0;
    for(NODE *iter = xcsf->pset; iter != NULL; iter = iter->next) {
        sum += cl_del_vote(xcsf, iter->cl, avg_fit);
    }
    double p = drand() * sum;

    // find the classifier to delete using the point
    sum = 0.0;
    NODE *prev = NULL;
    for(NODE *iter = xcsf->pset; iter != NULL; iter = iter->next) {
        sum += cl_del_vote(xcsf, iter->cl, avg_fit);
        if(sum > p) {
            iter->cl->num--;
            xcsf->pop_num_sum--;
            // macro classifier must be deleted
            if(iter->cl->num == 0) {
                set_add(xcsf, kset, iter->cl);
                xcsf->pop_num--;
                if(prev == NULL) {
                    xcsf->pset = iter->next;
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

void pop_enforce_limit(XCSF *xcsf, NODE **kset)
{
    while(xcsf->pop_num_sum > xcsf->POP_SIZE) {
        pop_del(xcsf, kset);
    }
}

void set_match(XCSF *xcsf, NODE **set, int *size, int *num, double *x, int time, NODE **kset)
{
    // add classifiers that match the input state to the match set  
#ifdef PARALLEL_MATCH
    NODE *blist[xcsf->pop_num];
    int j = 0;
    for(NODE *iter = xcsf->pset; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
    // update current matching conditions
    int s = 0; int n = 0;
#pragma omp parallel for reduction(+:s,n)
    for(int i = 0; i < xcsf->pop_num; i++) {
        if(cl_match(xcsf, blist[i]->cl, x)) {
            s++;
            n += blist[i]->cl->num;
        }
    }
    *size = s; *num = n;
    // build m list
    for(int i = 0; i < xcsf->pop_num; i++) {
        if(cl_match_state(xcsf, blist[i]->cl)) {
            set_add(xcsf, set, blist[i]->cl);
        }
    }
#else
    for(NODE *iter = xcsf->pset; iter != NULL; iter = iter->next) {
        if(cl_match(xcsf, iter->cl, x)) {
            set_add(xcsf, set, iter->cl);
            *num += iter->cl->num;
            (*size)++;                    
        }
    }   
#endif
    // perform covering if match set size is < THETA_MNA
    while(*size < xcsf->THETA_MNA) {
        // new classifier with matching condition
        CL *new = malloc(sizeof(CL));
        cl_init(xcsf, new, *num+1, time);
        cl_cover(xcsf, new, x);
        (*size)++;
        (*num)++;
        pop_add(xcsf, new);
        set_add(xcsf, set, new); 
        pop_enforce_limit(xcsf, kset);
        // remove any deleted classifiers from the match set
        set_validate(xcsf, set, size, num);
    }
}

void set_pred(XCSF *xcsf, NODE **set, int size, double *x, double *y)
{
    // match set fitness weighted prediction
    double *presum = calloc(xcsf->num_y_vars, sizeof(double));
    double fitsum = 0.0;
#ifdef PARALLEL_PRED
    NODE *blist[size];
    int j = 0;
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
#pragma omp parallel for reduction(+:presum[:xcsf->num_y_vars],fitsum)
    for(int i = 0; i < size; i++) {
        double *predictions = cl_predict(xcsf, blist[i]->cl, x);
        for(int var = 0; var < xcsf->num_y_vars; var++) {
            presum[var] += predictions[var] * blist[i]->cl->fit;
        }
        fitsum += blist[i]->cl->fit;
    }
#else
    (void)size; // remove unused parameter warnings
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        double *predictions = cl_predict(xcsf, iter->cl, x);
        for(int var = 0; var < xcsf->num_y_vars; var++) {
            presum[var] += predictions[var] * iter->cl->fit;
        }
        fitsum += iter->cl->fit;
    }    
#endif
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        y[var] = presum[var]/fitsum;
    }
    // clean up
    free(presum);
}

void set_add(XCSF *xcsf, NODE **set, CL *c)
{
    // adds a classifier to the set
    if(*set == NULL) {
        *set = malloc(sizeof(NODE));
        (*set)->cl = c;
        (*set)->next = NULL;
    }
    else {
        NODE *new = malloc(sizeof(NODE));
        new->cl = c;
        new->next = *set;
        *set = new;
    }
	(void)xcsf;
}

void set_update(XCSF *xcsf, NODE **set, int *size, int *num, double *y, NODE **kset, double *x)
{
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        cl_update(xcsf, iter->cl, x, y, *num);
    }
    set_update_fit(xcsf, set, *size, *num);
    if(xcsf->SET_SUBSUMPTION) {
        set_subsumption(xcsf, set, size, num, kset);
    }
}

void set_update_fit(XCSF *xcsf, NODE **set, int size, int num_sum)
{
    double acc_sum = 0.0;
    double accs[size];
    // calculate accuracies
    int i = 0;
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        accs[i] = cl_acc(xcsf, iter->cl);
        acc_sum += accs[i] * num_sum;
        i++;
    }
    // update fitnesses
    i = 0;
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        cl_update_fit(xcsf, iter->cl, acc_sum, accs[i]);
        i++;
    }
}

void set_subsumption(XCSF *xcsf, NODE **set, int *size, int *num, NODE **kset)
{
    CL *s = NULL;
    NODE *iter;
    // find the most general subsumer in the set
    for(iter = *set; iter != NULL; iter = iter->next) {
        CL *c = iter->cl;
        if(cl_subsumer(xcsf, c)) {
            if(s == NULL || cl_general(xcsf, c, s)) {
                s = c;
            }
        }
    }
    // subsume the more specific classifiers in the set
    if(s != NULL) {
        iter = *set; 
        while(iter != NULL) {
            CL *c = iter->cl;
            iter = iter->next;
            if(cl_general(xcsf, s, c)) {
                s->num += c->num;
                c->num = 0;
                set_add(xcsf, kset, c);
                set_validate(xcsf, set, size, num);
                set_validate(xcsf, &xcsf->pset, &xcsf->pop_num, &xcsf->pop_num_sum);
            }
        }
    }
}

void set_validate(XCSF *xcsf, NODE **set, int *size, int *num)
{
    // remove nodes pointing to classifiers with 0 numerosity
    *size = 0;
    *num = 0;
    NODE *prev = NULL;
    NODE *iter = *set;
    while(iter != NULL) {
        if(iter->cl == NULL || iter->cl->num == 0) {
            if(prev == NULL) {
                *set = iter->next;
                free(iter);
                iter = *set;
            }
            else {
                prev->next = iter->next;
                free(iter);
                iter = prev->next;
            }
        }
        else {
            (*size)++;
            (*num) += iter->cl->num;
            prev = iter;
            iter = iter->next;
        }
    }
	(void)xcsf;
}

void set_print(XCSF *xcsf, NODE *set)
{
    for(NODE *iter = set; iter != NULL; iter = iter->next) {
        cl_print(xcsf, iter->cl);
    }
	(void)xcsf;
}

void set_times(XCSF *xcsf, NODE **set, int time)
{
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        iter->cl->time = time;
    }
	(void)xcsf;
}

double set_total_fit(XCSF *xcsf, NODE **set)
{
    double sum = 0.0;
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        sum += iter->cl->fit;
    }
	(void)xcsf;
    return sum;
}

double set_total_time(XCSF *xcsf, NODE **set)
{
    double sum = 0.0;
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        sum += iter->cl->time * iter->cl->num;
    }
	(void)xcsf;
    return sum;
}

double set_mean_time(XCSF *xcsf, NODE **set, int num_sum)
{
    return set_total_time(xcsf, set) / num_sum;
}

void set_free(XCSF *xcsf, NODE **set)
{
    // frees the set only, not the classifiers
    NODE *iter = *set;
    while(iter != NULL) {
        *set = iter->next;
        free(iter);
        iter = *set;
    }
	(void)xcsf;
}

void set_kill(XCSF *xcsf, NODE **set)
{
    // frees the set and classifiers
    NODE *iter = *set;
    while(iter != NULL) {
        cl_free(xcsf, iter->cl);
        *set = iter->next;
        free(iter);
        iter = *set;
    }
}

double set_avg_mut(XCSF *xcsf, NODE **set, int m)
{
    // returns the average classifier mutation rate
    double sum = 0.0;
    int cnt = 0;
    for(NODE *iter = *set; iter != NULL; iter = iter->next) {
        sum += cl_mutation_rate(xcsf, iter->cl, m);
        cnt++;
    }
    return sum/cnt;
}
