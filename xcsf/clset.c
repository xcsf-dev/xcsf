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
 * @file clset.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Functions operating on sets of classifiers.
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "clset.h"

#include "prediction.h"
#include "neural.h"
#include "pred_neural.h"

#define MAX_COVER 1000000 //!< maximum number of covering attempts

static _Bool clset_action_coverage(const XCSF *xcsf, _Bool *act_covered);
static double clset_total_time(const SET *set);
static void clset_cover(XCSF *xcsf, const double *x);
static void clset_pop_del(XCSF *xcsf);
static void clset_pop_never_match(const XCSF *xcsf, CLIST **del, CLIST **delprev);
static void clset_pop_roulette(const XCSF *xcsf, CLIST **del, CLIST **delprev);
static void clset_subsumption(XCSF *xcsf, SET *set);
static void clset_update_fit(const XCSF *xcsf, const SET *set);

/**
 * @brief Initialises a new population of random classifiers.
 * @param xcsf The XCSF data structure.
 */
void clset_pop_init(XCSF *xcsf)
{
    if(xcsf->POP_INIT) {
        while(xcsf->pset.num < xcsf->POP_SIZE) {
            CL *new = malloc(sizeof(CL));
            cl_init(xcsf, new, xcsf->POP_SIZE, 0);
            cl_rand(xcsf, new);
            clset_add(&xcsf->pset, new);
        }
    }
}

/**
 * @brief Initialises a new set.
 * @param set The set to be initialised.
 */
void clset_init(SET *set)
{
    set->list = NULL;
    set->size = 0;
    set->num = 0;
}

/**
 * @brief Deletes a single classifier from the population set.
 * @param xcsf The XCSF data structure.
 */
static void clset_pop_del(XCSF *xcsf)
{
    CLIST *del = NULL;
    CLIST *delprev = NULL;
    // select any rules that never match
    clset_pop_never_match(xcsf, &del, &delprev);
    // if none found, select a rule using roulette wheel
    if(del == NULL) {
        clset_pop_roulette(xcsf, &del, &delprev);
    }
    // decrement numerosity
    (del->cl->num)--;
    (xcsf->pset.num)--;
    // macro classifier must be deleted
    if(del->cl->num == 0) {
        clset_add(&xcsf->kset, del->cl);
        (xcsf->pset.size)--;
        if(delprev == NULL) {
            xcsf->pset.list = del->next;
        }
        else {
            delprev->next = del->next;
        }
        free(del);
    }
}

/**
 * @brief Finds a rule in the population that never matches an input.
 * @param xcsf The XCSF data structure.
 * @param del A pointer to the classifier to be deleted (set by this function).
 * @param delprev A pointer to the rule previous to the one being deleted (set by this function).
 */
static void clset_pop_never_match(const XCSF *xcsf, CLIST **del, CLIST **delprev)
{
    CLIST *prev = NULL;
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        if(iter->cl->mtotal == 0 && iter->cl->age > xcsf->M_PROBATION) {
            *del = iter;
            *delprev = prev;
            break;
        }
        prev = iter;
    }
}

/*
 * @brief Selects a classifier from the population for deletion via roulette wheel.
 * @param xcsf The XCSF data structure.
 * @param del A pointer to the rule to be deleted (set by this function).
 * @param delprev A pointer to the rule previous to the one being deleted (set by this function).
 *
 * @details Two classifiers are selected using roulette wheel selection with the
 * deletion vote and the one with the largest condition + prediction size is
 * chosen. For fixed-length representations the effect is the same as one
 * roulete spin.
 */
static void clset_pop_roulette(const XCSF *xcsf, CLIST **del, CLIST **delprev)
{
    double avg_fit = clset_total_fit(&xcsf->pset) / xcsf->pset.num;
    double total_vote = 0;
    for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        total_vote += cl_del_vote(xcsf, iter->cl, avg_fit);
    }
    int delsize = 0;
    for(int i = 0; i < 2; i++) {
        // perform a single roulette spin with the deletion vote
        CLIST *iter = xcsf->pset.list;
        CLIST *prev = NULL;
        double p = rand_uniform(0, total_vote);
        double sum = cl_del_vote(xcsf, iter->cl, avg_fit);
        while(p > sum) {
            prev = iter;
            iter = iter->next;
            sum += cl_del_vote(xcsf, iter->cl, avg_fit);
        }
        // select the rule for deletion if it is the largest sized winner
        int size = cl_cond_size(xcsf, iter->cl) + cl_pred_size(xcsf, iter->cl);
        if(*del == NULL || size > delsize) {
            *del = iter;
            *delprev = prev;
            delsize = size;
        }
    }
}

/**
 * @brief Enforces the maximum population size limit.
 * @param xcsf The XCSF data structure.
 */ 
void clset_pop_enforce_limit(XCSF *xcsf)
{
    while(xcsf->pset.num > xcsf->POP_SIZE) {
        clset_pop_del(xcsf);
    }
}

/**
 * @brief Constructs the match set.
 * @param xcsf The XCSF data structure.
 * @param x The input state.
 * @details Processes the matching conditions for each classifier in the
 * population. If a classifier matches, its action is updated and it is added
 * to the match set. Covering is performed if any actions are unrepresented.
 */
void clset_match(XCSF *xcsf, const double *x)
{
#ifdef PARALLEL_MATCH
    // prepare for parallel processing of matching conditions
    CLIST *blist[xcsf->pset.size];
    int j = 0;
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
    // update current matching conditions setting m flags in parallel
#pragma omp parallel for
    for(int i = 0; i < xcsf->pset.size; i++) {
        cl_match(xcsf, blist[i]->cl, x);
    }
    // build match set list in series
    for(int i = 0; i < xcsf->pset.size; i++) {
        if(cl_m(xcsf, blist[i]->cl)) {
            clset_add(&xcsf->mset, blist[i]->cl);
            cl_action(xcsf, blist[i]->cl, x);
        }
    }
#else
    // update matching conditions and build match set list in series
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        if(cl_match(xcsf, iter->cl, x)) {
            clset_add(&xcsf->mset, iter->cl);
            cl_action(xcsf, iter->cl, x);
        }
    }   
#endif
    // perform covering if all actions are not represented
    if(xcsf->n_actions > 1 || xcsf->mset.size < 1) {
        clset_cover(xcsf, x);
    }
    // update statistics
    xcsf->msetsize += (xcsf->mset.size - xcsf->msetsize) * (10 / (double) xcsf->PERF_TRIALS);
    xcsf->mfrac += (clset_mfrac(xcsf) - xcsf->mfrac) * (10 / (double) xcsf->PERF_TRIALS);
}

/**
 * @brief Ensures all possible actions are covered by the match set.
 * @param xcsf The XCSF data structure.
 * @param x The input state.
 */
static void clset_cover(XCSF *xcsf, const double *x)
{
    int attempts = 0;
    _Bool *act_covered = malloc(xcsf->n_actions * sizeof(_Bool));
    _Bool covered = clset_action_coverage(xcsf, act_covered);
    while(!covered) {
        covered = true;
        for(int i = 0; i < xcsf->n_actions; i++) {
            if(!act_covered[i]) {
                // new classifier with matching condition and action
                CL *new = malloc(sizeof(CL));
                cl_init(xcsf, new, (xcsf->mset.num)+1, xcsf->time);
                cl_cover(xcsf, new, x, i);
                clset_add(&xcsf->pset, new);
                clset_add(&xcsf->mset, new);
            }
        }
        // enforce population size
        int prev_psize = xcsf->pset.size;
        clset_pop_enforce_limit(xcsf);
        // if a macro classifier was deleted, remove any deleted rules from the match set
        if(prev_psize > xcsf->pset.size) {
            int prev_msize = xcsf->mset.size;
            clset_validate(&xcsf->mset);
            // if the deleted classifier was in the match set,
            // check if an action is now not covered
            if(prev_msize > xcsf->mset.size) {
                covered = clset_action_coverage(xcsf, act_covered);
            }
        }
        attempts++;
        if(attempts > MAX_COVER) {
            printf("Error: maximum covering attempts (%d) exceeded\n", MAX_COVER);
            exit(EXIT_FAILURE);
        }
    }
    free(act_covered);
}

/**
 * @brief Checks whether each action is covered by the match set.
 * @param xcsf The XCSF data structure.
 * @param act_covered Array of action coverage flags (set by this function).
 * @return Whether all actions are covered.
 */
static _Bool clset_action_coverage(const XCSF *xcsf, _Bool *act_covered)
{
    memset(act_covered, 0, xcsf->n_actions * sizeof(_Bool));
    for(const CLIST *iter = xcsf->mset.list; iter != NULL; iter = iter->next) {
        act_covered[iter->cl->action] = true;
    }
    for(int i = 0; i < xcsf->n_actions; i++) {
        if(!act_covered[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Calculates the set mean fitness weighted prediction.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the prediction.
 * @param x The input state.
 * @param p The predictions (set by this function).
 */
void clset_pred(const XCSF *xcsf, const SET *set, const double *x, double *p)
{
    double *presum = calloc(xcsf->y_dim, sizeof(double));
    double fitsum = 0;
#ifdef PARALLEL_PRED
    CLIST *blist[set->size];
    int j = 0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
#pragma omp parallel for reduction(+:presum[:xcsf->y_dim],fitsum)
    for(int i = 0; i < set->size; i++) {
        const double *predictions = cl_predict(xcsf, blist[i]->cl, x);
        for(int var = 0; var < xcsf->y_dim; var++) {
            presum[var] += predictions[var] * blist[i]->cl->fit;
        }
        fitsum += blist[i]->cl->fit;
    }
#pragma omp parallel for
    for(int var = 0; var < xcsf->y_dim; var++) {
        p[var] = presum[var] / fitsum;
    }
#else
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        const double *predictions = cl_predict(xcsf, iter->cl, x);
        for(int var = 0; var < xcsf->y_dim; var++) {
            presum[var] += predictions[var] * iter->cl->fit;
        }
        fitsum += iter->cl->fit;
    }    
    for(int var = 0; var < xcsf->y_dim; var++) {
        p[var] = presum[var] / fitsum;
    }
#endif
    free(presum);
}    

/**
 * @brief Constructs the action set from the match set.
 * @param xcsf The XCSF data structure.
 * @param action The action used to build the set.
 */
void clset_action(XCSF *xcsf, int action)
{
    for(const CLIST *iter = xcsf->mset.list; iter != NULL; iter = iter->next) {
        if(iter->cl->action == action) {
            clset_add(&xcsf->aset, iter->cl);
        }
    }   
}        

/**
 * @brief Adds a classifier to the set.
 * @param set The set to add the classifier.
 * @param c The classifier to add.
 */
void clset_add(SET *set, CL *c)
{
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

/**
 * @brief Provides reinforcement to the set and performs set subsumption.
 * @param xcsf The XCSF data structure.
 * @param set The set to provide reinforcement.
 * @param x The input state.
 * @param y The payoff from the environment.
 * @param cur Whether the update is for the current or previous state.
 */ 
void clset_update(XCSF *xcsf, SET *set, const double *x, const double *y, _Bool cur)
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
        cl_update(xcsf, blist[i]->cl, x, y, set->num, cur);
    }
#else
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        cl_update(xcsf, iter->cl, x, y, set->num, cur);
    }
#endif
    clset_update_fit(xcsf, set);
    if(xcsf->SET_SUBSUMPTION) {
        clset_subsumption(xcsf, set);
    }
}

/**
 * @brief Updates the fitness of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to update.
 */ 
static void clset_update_fit(const XCSF *xcsf, const SET *set)
{
    double acc_sum = 0;
    double accs[set->size];
    // calculate accuracies
    int i = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        accs[i] = cl_acc(xcsf, iter->cl);
        acc_sum += accs[i] * iter->cl->num;
        i++;
    }
    // update fitnesses
    i = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        cl_update_fit(xcsf, iter->cl, acc_sum, accs[i]);
        i++;
    }
}

/**
 * @brief Performs set subsumption.
 * @param xcsf The XCSF data structure.
 * @param set The set to perform subsumption.
 */ 
static void clset_subsumption(XCSF *xcsf, SET *set)
{
    // find the most general subsumer in the set
    CL *s = NULL;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        CL *c = iter->cl;
        if(cl_subsumer(xcsf, c) && (s == NULL || cl_general(xcsf, c, s))) {
            s = c;
        }
    }
    // subsume the more specific classifiers in the set
    if(s != NULL) {
        const CLIST *iter = set->list;
        while(iter != NULL) {
            CL *c = iter->cl;
            iter = iter->next;
            if(s != c && cl_general(xcsf, s, c)) {
                s->num += c->num;
                c->num = 0;
                clset_add(&xcsf->kset, c);
                clset_validate(set);
                clset_validate(&xcsf->pset);
            }
        }
    }
}

/**
 * @brief Removes classifiers with 0 numerosity from the set.
 * @param set The set to validate.
 */ 
void clset_validate(SET *set)
{
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

/**
 * @brief Prints the classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to print.
 * @param printc Whether to print the conditions.
 * @param printa Whether to print the actions.
 * @param printp Whether to print the predictions.
 */
void clset_print(const XCSF *xcsf, const SET *set, _Bool printc, _Bool printa, _Bool printp)
{
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        cl_print(xcsf, iter->cl, printc, printa, printp);
    }
}

/**
 * @brief Sets the time stamps for classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to update the time stamps.
 */
void clset_set_times(const XCSF *xcsf, const SET *set)
{
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        iter->cl->time = xcsf->time;
    }
}

/**
 * @brief Calculates the total fitness of classifiers in the set.
 * @param set The set to calculate the total fitness.
 * @return The total fitness of classifiers in the set.
 */ 
double clset_total_fit(const SET *set)
{
    double sum = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += iter->cl->fit;
    }
    return sum;
}

/**
 * @brief Calculates the total time stamps of classifiers in the set.
 * @param set The set to calculate the total time.
 * @return The total time of classifiers in the set.
 */ 
static double clset_total_time(const SET *set)
{
    double sum = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += iter->cl->time * iter->cl->num;
    }
    return sum;
}

/**
 * @brief Calculates the mean time stamp of classifiers in the set.
 * @param set The set to calculate the mean time.
 * @return The mean time of classifiers in the set.
 */ 
double clset_mean_time(const SET *set)
{
    return clset_total_time(set) / set->num;
}

/**
 * @brief Frees the set, but not the classifiers.
 * @param set The set to free.
 */ 
void clset_free(SET *set)
{
    CLIST *iter = set->list;
    while(iter != NULL) {
        set->list = iter->next;
        free(iter);
        iter = set->list;
    }
    set->size = 0;
    set->num = 0;
}

/**
 * @brief Frees the set and the classifiers.
 * @param xcsf The XCSF data structure.
 * @param set The set to free.
 */ 
void clset_kill(const XCSF *xcsf, SET *set)
{
    CLIST *iter = set->list;
    while(iter != NULL) {
        cl_free(xcsf, iter->cl);
        set->list = iter->next;
        free(iter);
        iter = set->list;
    }
    set->size = 0;
    set->num = 0;
}

/**
 * @brief Writes the population set to a binary file.
 * @param xcsf The XCSF data structure.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t clset_pop_save(const XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&xcsf->pset.size, sizeof(int), 1, fp);
    s += fwrite(&xcsf->pset.num, sizeof(int), 1, fp);
    for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        s += cl_save(xcsf, iter->cl, fp);
    }
    return s;
}

/**
 * @brief Reads the population set from a binary file.
 * @param xcsf The XCSF data structure.
 * @param fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t clset_pop_load(XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    int size = 0;
    int num = 0;
    s += fread(&size, sizeof(int), 1, fp);
    s += fread(&num, sizeof(int), 1, fp);
    clset_init(&xcsf->pset);
    for(int i = 0; i < size; i++) {
        CL *c = malloc(sizeof(CL));
        s += cl_load(xcsf, c, fp);
        clset_add(&xcsf->pset, c);
    }
    return s;
}

/**
 * @brief Calculates the mean condition size of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean condition size.
 * @return The mean condition size of classifiers in the set.
 */
double clset_mean_cond_size(const XCSF *xcsf, const SET *set)
{
    int sum = 0;
    int cnt = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += cl_cond_size(xcsf, iter->cl);
        cnt++;
    }
    return (double) sum / cnt;
}

/**
 * @brief Calculates the mean prediction size of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean prediction size.
 * @return The mean prediction size of classifiers in the set.
 */ 
double clset_mean_pred_size(const XCSF *xcsf, const SET *set)
{
    int sum = 0;
    int cnt = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += cl_pred_size(xcsf, iter->cl);
        cnt++;
    }
    return (double) sum / cnt;
}

/**
 * @brief Returns the fraction of inputs matched by the most general rule with
 * error below EPS_0. If no rules below EPS_0, the lowest error rule is used.
 * @param xcsf The XCSF data structure.
 * @return The fraction of inputs matched.
 */ 
double clset_mfrac(const XCSF *xcsf)
{
    double mfrac = 0;
    // most general rule below EPS_0
    for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        double e = iter->cl->err;
        if(e < xcsf->EPS_0 && iter->cl->exp > 1 / xcsf->BETA) {
            double m = cl_mfrac(xcsf, iter->cl);
            if(m > mfrac) {
                mfrac = m;
            }
        }
    }
    // lowest error rule
    if(mfrac == 0) {
        double error = DBL_MAX;
        for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
            double e = iter->cl->err;
            if(e < error && iter->cl->exp > 1 / xcsf->BETA) {
                mfrac = cl_mfrac(xcsf, iter->cl);
                error = e;
            }
        }
    }
    return mfrac;
}

/* Neural network prediction functions */

/**
 * @brief Calculates the mean prediction layer ETA of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean.
 * @param layer The position of layer to calculate.
 * @return The mean prediction layer ETA of classifiers in the set.
 */ 
double clset_mean_eta(const XCSF *xcsf, const SET *set, int layer)
{
    double sum = 0;
    int cnt = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += pred_neural_eta(xcsf, iter->cl, layer);
        cnt++;
    }
    return sum / cnt;
}

/**
 * @brief Calculates the mean number of prediction neurons for a given layer.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean.
 * @param layer The neural network layer position.
 * @return The mean number of neurons in the layer.
 */
double clset_mean_neurons(const XCSF *xcsf, const SET *set, int layer)
{
    int sum = 0;
    int cnt = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += pred_neural_neurons(xcsf, iter->cl, layer);
        cnt++;
    }
    return (double) sum / cnt;
}

/**
 * @brief Calculates the mean number of prediction layers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean.
 * @return The mean number of layers.
 */
double clset_mean_layers(const XCSF *xcsf, const SET *set)
{
    int sum = 0;
    int cnt = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += pred_neural_layers(xcsf, iter->cl);
        cnt++;
    }
    return (double) sum / cnt;
}
