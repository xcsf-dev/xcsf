/*
 * Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
#include "cons.h"
#include "random.h"
#include "cl.h"
#include "cl_set.h"

void set_subsumption(NODE **set, int *size, int *num, NODE **kset);
void set_update_fit(NODE **set, int size, int num);

void pop_init()
{
	// initialise population
	pset = NULL; // population linked list
	pop_num = 0; // num macro-classifiers
	pop_num_sum = 0; // numerosity sum

	if(POP_INIT) {
		while(pop_num_sum < POP_SIZE) {
			CL *new = malloc(sizeof(CL));
			cl_init(new, POP_SIZE, 0);
			cond_rand(&new->cond);
			pop_add(new);
		}
	}
}
            
void pop_add(CL *c)
{
	// adds a classifier to the population set
	pop_num_sum++;
	pop_num++;
	if(pset == NULL) {
		pset = malloc(sizeof(NODE));
		pset->cl = c;
		pset->next = NULL;
	}
	else {
		NODE *new = malloc(sizeof(NODE));
		new->next = pset;
		new->cl = c;
		pset = new;
	}
}

NODE *pop_del()
{
	// selects a classifier using roullete wheel selection with the deletion 
	// vote; sets its numerosity to zero, and removes it from the population 
	// set; returns the classifier to be deleted.
	NODE *iter;
	NODE *prev = NULL;
	double avg_fit = set_total_fit(&pset) / pop_num_sum;
	double sum = 0.0;
	for(iter = pset; iter != NULL; iter = iter->next)
		sum += cl_del_vote(iter->cl, avg_fit);
	double p = drand() * sum;
	sum = 0.0;
	for(iter=pset; iter != NULL; prev=iter, iter=iter->next) {
		sum += cl_del_vote(iter->cl, avg_fit);
		if(sum > p) {
			iter->cl->num--;
			pop_num_sum--;
			// macro classifier must be deleted
			if(iter->cl->num == 0) {
				pop_num--;
				if(prev == NULL)
					pset = iter->next;
				else
					prev->next = iter->next;
			}
			return iter;
		}
	}   
	return iter;
}
 
void pop_enforce_limit(NODE **kset)
{
 	while(pop_num_sum > POP_SIZE) {
		NODE *del = pop_del();
		if(del->cl->num == 0)
			set_add(kset, del->cl);
	}     
}
 
void set_match(NODE **set, int *size, int *num, double *state, int time, NODE **kset)
{
	// find matching classifiers in the population
	for(NODE *iter = pset; iter != NULL; iter = iter->next) {
		if(cond_match(&iter->cl->cond, state)) {
			set_add(set, iter->cl);
			*num += iter->cl->num;
			(*size)++;
		}
	}   
	// perform covering if match set is < THETA_MNA
	while(*size < THETA_MNA) {
		// new classifier with matching condition
		CL *new = malloc(sizeof(CL));
		cl_init(new, *num+1, time);
		cond_cover(&new->cond, state);
		(*size)++;
		(*num)++;
		pop_add(new);
		set_add(set, new); 
		pop_enforce_limit(kset);
		// remove any deleted classifiers from the match set
		set_validate(set, size, num);
	}
}

double set_pred(NODE **set, double *state)
{
	// fitness weighted prediction
	double presum = 0.0;
	double fitsum = 0.0;
	for(NODE *iter = *set; iter != NULL; iter = iter->next) {
		presum += pred_compute(&iter->cl->pred, state) * iter->cl->fit;
		fitsum += iter->cl->fit;
	}
	return presum/fitsum;
}

void set_add(NODE **set, CL *c)
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
}

void set_update(NODE **set, int *size, int *num, double r, NODE **kset, double *state)
{
	for(NODE *iter = *set; iter != NULL; iter = iter->next)
		cl_update(iter->cl, state, r, *num);
	set_update_fit(set, *size, *num);
	if(SET_SUBSUMPTION)
		set_subsumption(set, size, num, kset);
}

void set_update_fit(NODE **set, int size, int num_sum)
{
	double acc_sum = 0.0;
	double accs[size];
	// calculate accuracies
	int i = 0;
	for(NODE *iter = *set; iter != NULL; iter = iter->next) {
		accs[i] = cl_acc(iter->cl);
		acc_sum += accs[i] * num_sum;
		i++;
	}
	// update fitnesses
	i = 0;
	for(NODE *iter = *set; iter != NULL; iter = iter->next) {
		cl_update_fit(iter->cl, acc_sum, accs[i]);
		i++;
	}
}

void set_subsumption(NODE **set, int *size, int *num, NODE **kset)
{
	CL *s = NULL;
	NODE *iter;
	// find the most general subsumer in the set
	for(iter = *set; iter != NULL; iter = iter->next) {
		CL *c = iter->cl;
		if(cl_subsumer(c)) {
			if(s == NULL || cond_general(&(c->cond), &(s->cond)))
				s = c;
		}
	}
	// subsume the more specific classifiers in the set
	if(s != NULL) {
		iter = *set; 
		while(iter != NULL) {
			CL *c = iter->cl;
			if(cond_general(&(s->cond), &(c->cond))) {
				s->num += c->num;
				c->num = 0;
				set_add(kset, c);
				set_validate(set, size, num);
				set_validate(&pset, &pop_num, &pop_num_sum);
			}
			iter = iter->next;
		}
	}
}

void set_validate(NODE **set, int *size, int *num)
{
	// remove nodes pointing to classifiers with 0 numerosity
	*size = 0;
	*num = 0;
	NODE *prev = NULL;
	NODE *iter = *set;
	while(iter != NULL) {
		if(iter->cl == NULL || iter->cl->num == 0) {
			if(prev == NULL)
				*set = iter->next;
			else
				prev->next = iter->next;
			iter = iter->next;
			continue;
		}
		else {
			(*size)++;
			(*num) += iter->cl->num;
		}
		prev = iter;
		iter = iter->next;
	}
}

void set_print(NODE *set)
{
	for(NODE *iter = set; iter != NULL; iter = iter->next)
		cl_print(iter->cl);
}

void set_times(NODE **set, int time)
{
	for(NODE *iter = *set; iter != NULL; iter = iter->next)
		iter->cl->time = time;
}

double set_total_fit(NODE **set)
{
	double sum = 0.0;
	for(NODE *iter = *set; iter != NULL; iter = iter->next)
		sum += iter->cl->fit;
	return sum;
}

double set_total_time(NODE **set)
{
	double sum = 0.0;
	for(NODE *iter = *set; iter != NULL; iter = iter->next)
		sum += iter->cl->time * iter->cl->num;
	return sum;
}

double set_mean_time(NODE **set, int num_sum)
{
	return set_total_time(set) / num_sum;
}

void set_free(NODE **set)
{
	// frees the set only, not the classifiers
	NODE *iter = *set;
	while(iter != NULL) {
		*set = iter->next;
		free(iter);
		iter = *set;
	}
}

void set_kill(NODE **set)
{
	// frees the set and classifiers
	NODE *iter = *set;
	while(iter != NULL) {
		cl_free(iter->cl);
		*set = iter->next;
		free(iter);
		iter = *set;
	}
}

void set_clean(NODE **kset, NODE **set, _Bool in_set)
{
	// if in_set = false, removes classifiers from kset
	// that are *not* in the set; otherwise removes only
	// classifiers from kset that *are* in the set
	NODE *iter, *kiter, *prev_kiter = NULL;
	for(iter = *set; iter != NULL; iter = iter->next) {
		for(kiter = *kset; kiter != NULL; kiter = kiter->next) {
			if((!in_set && iter->cl != kiter->cl) ||
					(in_set && iter->cl == kiter->cl)) {
				if(prev_kiter == NULL)
					*kset = kiter->next;
				else
					prev_kiter->next = kiter->next;
				cl_free(kiter->cl);
				free(kiter);
			}
		}
	}
}

#ifdef SELF_ADAPT_MUTATION
double set_avg_mut(NODE **set, int m)
{
	// returns the average classifier mutation rate
	double sum = 0.0;
	int cnt = 0;
	for(NODE *iter = *set; iter != NULL; iter = iter->next) {
		sum += iter->cl->cond.mu[m];
		cnt++;
	}
	return sum/cnt;
}
#endif
