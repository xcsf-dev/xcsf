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
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "clset.h"

#include "prediction.h"
#include "pred_neural.h"

#define MAX_COVER 1000000 //!< maximum number of covering attempts

static _Bool clset_action_covered(const XCSF *xcsf, const SET *set, int action);
static double clset_total_time(const XCSF *xcsf, const SET *set);
static void clset_cover(XCSF *xcsf, SET *mset, SET *kset, const double *x, _Bool *act_covered);
static void clset_pop_del(XCSF *xcsf, SET *kset);
static void clset_subsumption(XCSF *xcsf, SET *set, SET *kset);
static void clset_update_fit(const XCSF *xcsf, const SET *set);

/**
 * @brief Initialises a new population set.
 * @param xcsf The XCSF data structure.
 */
void clset_pop_init(XCSF *xcsf)
{
    xcsf->time = 0; // number of learning trials performed
    xcsf->msetsize = 0; // average match set size
    clset_init(xcsf, &xcsf->pset);
    // initialise population with random classifiers
    if(xcsf->POP_INIT) {
        while(xcsf->pset.num < xcsf->POP_SIZE) {
            CL *new = malloc(sizeof(CL));
            cl_init(xcsf, new, xcsf->POP_SIZE, 0);
            cl_rand(xcsf, new);
            clset_add(xcsf, &xcsf->pset, new);
        }
    }
}

/**
 * @brief Initialises a new set.
 * @param xcsf The XCSF data structure.
 * @param set The set to be initialised.
 */
void clset_init(const XCSF *xcsf, SET *set)
{
    (void)xcsf;
    set->list = NULL;
    set->size = 0;
    set->num = 0;
}

/**
 * @brief Deletes a single classifier from the population set.
 * @param xcsf The XCSF data structure.
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 *
 * @details Selects two classifiers using roulete wheel selection with the
 * deletion vote and deletes the one with the largest condition + prediction
 * length. For fixed-length representations this is the same as one roulete spin.
 */
static void clset_pop_del(XCSF *xcsf, SET *kset)
{
    double avg_fit = clset_total_fit(xcsf, &xcsf->pset) / xcsf->pset.num;
    double total = 0;
    for(const CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        total += cl_del_vote(xcsf, iter->cl, avg_fit);
    }
    CLIST *del = NULL;
    CLIST *delprev = NULL;
    int delsize = 0;
    for(int i = 0; i < 2; i++) {
        double p = rand_uniform(0,total);
        double sum = 0;
        CLIST *prev = NULL;
        for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
            sum += cl_del_vote(xcsf, iter->cl, avg_fit);
            if(sum > p) {
                int size = cl_cond_size(xcsf, iter->cl) + cl_pred_size(xcsf, iter->cl);
                if(del == NULL || size > delsize) {
                    del = iter;
                    delprev = prev;
                    delsize = size;
                }
                break;
            }
            prev = iter;
        }
    }
    // decrement numerosity
    (del->cl->num)--;
    (xcsf->pset.num)--;
    // macro classifier must be deleted
    if(del->cl->num == 0) {
        clset_add(xcsf, kset, del->cl);
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
 * @brief Enforces the maximum population size limit.
 * @param xcsf The XCSF data structure.
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 */ 
void clset_pop_enforce_limit(XCSF *xcsf, SET *kset)
{
    while(xcsf->pset.num > xcsf->POP_SIZE) {
        clset_pop_del(xcsf, kset);
    }
}

/**
 * @brief Constructs the match set.
 * @param xcsf The XCSF data structure.
 * @param mset The match set.
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 * @param x The input state.
 *
 * @details Processes the matching conditions for each classifier in the
 * population. Adds each matching classifier to the match set. Performs
 * covering if any actions are not covered.
 */
void clset_match(XCSF *xcsf, SET *mset, SET *kset, const double *x)
{
    _Bool *act_covered = calloc(xcsf->num_actions, sizeof(_Bool));
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
            clset_add(xcsf, mset, blist[i]->cl);
            int action = cl_action(xcsf, blist[i]->cl, x);
            act_covered[action] = true;
        }
    }
#else
    // update matching conditions and build match set list in series
    for(CLIST *iter = xcsf->pset.list; iter != NULL; iter = iter->next) {
        if(cl_match(xcsf, iter->cl, x)) {
            clset_add(xcsf, mset, iter->cl);
            int action = cl_action(xcsf, iter->cl, x);
            act_covered[action] = true;
        }
    }   
#endif
    // perform covering if all actions are not represented
    clset_cover(xcsf, mset, kset, x, act_covered);
    free(act_covered);
}

/**
 * @brief Ensures all possible actions are covered by the match set.
 * @param xcsf The XCSF data structure.
 * @param mset The match set.
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 * @param x The input state.
 * @param act_covered Array indicating whether each action is covered by the set.
 */
static void clset_cover(XCSF *xcsf, SET *mset, SET *kset, const double *x, _Bool *act_covered)
{
    int attempts = 0;
    _Bool again;
    do {
        again = false;
        for(int i = 0; i < xcsf->num_actions; i++) {
            if(!act_covered[i]) {
                // new classifier with matching condition and action
                CL *new = malloc(sizeof(CL));
                cl_init(xcsf, new, (mset->num)+1, xcsf->time);
                cl_cover(xcsf, new, x, i);
                clset_add(xcsf, &xcsf->pset, new);
                clset_add(xcsf, mset, new); 
                act_covered[i] = true;
            }
        }
        // enforce population size
        int prev_psize = xcsf->pset.size;
        clset_pop_enforce_limit(xcsf, kset);
        // if a macro classifier was deleted, remove any deleted rules from the match set
        if(prev_psize > xcsf->pset.size) {
            int prev_msize = mset->size;
            clset_validate(xcsf, mset);
            // if the deleted classifier was in the match set,
            // check if an action is now not covered
            if(prev_msize > mset->size) {
                for(int i = 0; i < xcsf->num_actions; i++) {
                    if(!clset_action_covered(xcsf, mset, i)) {
                        act_covered[i] = false;
                        again = true;
                    }
                }
            }
        }
        attempts++;
        if(attempts > MAX_COVER) {
            printf("Error: maximum covering attempts (%d) exceeded\n", MAX_COVER);
            exit(EXIT_FAILURE);
        }

    } while(again);
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
    double *presum = calloc(xcsf->num_y_vars, sizeof(double));
    double fitsum = 0;
#ifdef PARALLEL_PRED
    CLIST *blist[set->size];
    int j = 0;
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        blist[j] = iter;
        j++;
    }
#pragma omp parallel for reduction(+:presum[:xcsf->num_y_vars],fitsum)
    for(int i = 0; i < set->size; i++) {
        const double *predictions = cl_predict(xcsf, blist[i]->cl, x);
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
        const double *predictions = cl_predict(xcsf, iter->cl, x);
        for(int var = 0; var < xcsf->num_y_vars; var++) {
            presum[var] += predictions[var] * iter->cl->fit;
        }
        fitsum += iter->cl->fit;
    }    
    for(int var = 0; var < xcsf->num_y_vars; var++) {
        p[var] = presum[var]/fitsum;
    }
#endif
    free(presum);
}    

/**
 * @brief Constructs the action set.
 * @param xcsf The XCSF data structure.
 * @param mset The match set.
 * @param aset The action set.
 * @param action The action used to build the set.
 */
void clset_action(const XCSF *xcsf, const SET *mset, SET *aset, int action)
{
    for(const CLIST *iter = mset->list; iter != NULL; iter = iter->next) {
        if(iter->cl->action == action) {
            clset_add(xcsf, aset, iter->cl);
        }
    }   
}        

/**
 * @brief Returns whether an action is covered by the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to check (typically the match set).
 * @param action The action to check.
 * @return Whether the action is covered.
 */
static _Bool clset_action_covered(const XCSF *xcsf, const SET *set, int action)
{
    (void)xcsf;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        if(iter->cl->action == action) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Adds a classifier to the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to add the classifier.
 * @param c The classifier to add.
 */
void clset_add(const XCSF *xcsf, SET *set, CL *c)
{
    (void)xcsf;
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
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 * @param x The input state.
 * @param y The payoff from the environment.
 * @param curr Whether the update is for the current or previous state.
 */ 
void clset_update(XCSF *xcsf, SET *set, SET *kset, const double *x, const double *y, _Bool curr)
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
        cl_update(xcsf, blist[i]->cl, x, y, set->num, curr);
    }
#else
    for(CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        cl_update(xcsf, iter->cl, x, y, set->num, curr);
    }
#endif
    clset_update_fit(xcsf, set);
    if(xcsf->SET_SUBSUMPTION) {
        clset_subsumption(xcsf, set, kset);
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
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 */ 
static void clset_subsumption(XCSF *xcsf, SET *set, SET *kset)
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
                clset_add(xcsf, kset, c);
                clset_validate(xcsf, set);
                clset_validate(xcsf, &xcsf->pset);
            }
        }
    }
}

/**
 * @brief Removes classifiers with 0 numerosity from the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to validate.
 */ 
void clset_validate(const XCSF *xcsf, SET *set)
{
    (void)xcsf;
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
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the total fitness.
 * @return The total fitness of classifiers in the set.
 */ 
double clset_total_fit(const XCSF *xcsf, const SET *set)
{
    (void)xcsf;
    double sum = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += iter->cl->fit;
    }
    return sum;
}

/**
 * @brief Calculates the total time stamps of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the total time.
 * @return The total time of classifiers in the set.
 */ 
static double clset_total_time(const XCSF *xcsf, const SET *set)
{
    (void)xcsf;
    double sum = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += iter->cl->time * iter->cl->num;
    }
    return sum;
}

/**
 * @brief Calculates the mean time stamp of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean time.
 * @return The mean time of classifiers in the set.
 */ 
double clset_mean_time(const XCSF *xcsf, const SET *set)
{
    return clset_total_time(xcsf, set) / set->num;
}

/**
 * @brief Frees the set, but not the classifiers.
 * @param xcsf The XCSF data structure.
 * @param set The set to free.
 */ 
void clset_free(const XCSF *xcsf, SET *set)
{
    (void)xcsf;
    CLIST *iter = set->list;
    while(iter != NULL) {
        set->list = iter->next;
        free(iter);
        iter = set->list;
    }
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
    clset_init(xcsf, &xcsf->pset);
    for(int i = 0; i < size; i++) {
        CL *c = malloc(sizeof(CL));
        s += cl_load(xcsf, c, fp);
        clset_add(xcsf, &xcsf->pset, c);
    }
    return s;
}

/**
 * @brief Calculates the mean mutation rate of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean mutation rate.
 * @param m Which mutation rate to average.
 * @return The mean mutation rate of classifiers in the set.
 */ 
double clset_mean_mut(const XCSF *xcsf, const SET *set, int m)
{
    // return the fixed value if not adapted
    if(m >= xcsf->SAM_NUM) {
        switch(m) {
            case 0: return xcsf->S_MUTATION;
            case 1: return xcsf->P_MUTATION;
            case 2: return xcsf->E_MUTATION;
            case 3: return xcsf->F_MUTATION;
            default: return -1;
        }
    }
    // return the average classifier mutation rate
    double sum = 0;
    int cnt = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        sum += cl_mutation_rate(xcsf, iter->cl, m);
        cnt++;
    }
    return sum / cnt;
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
 * @brief Calculates the mean fraction of inputs matched by classifiers.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean prediction size.
 * @return The mean fraction of inputs matched.
 */ 
double clset_mean_inputs_matched(const XCSF *xcsf, const SET *set)
{
    double sum = 0;
    int cnt = 0;
    for(const CLIST *iter = set->list; iter != NULL; iter = iter->next) {
        if(iter->cl->exp > 1 / xcsf->BETA) {
            sum += cl_mfrac(xcsf, iter->cl);
            cnt++;
        }
    }
    if(cnt > 0) {
        return sum / cnt;
    }
    return 0;
}

/* Neural network prediction functions */

/**
 * @brief Calculates the mean prediction layer ETA of classifiers in the set.
 * @param xcsf The XCSF data structure.
 * @param set The set to calculate the mean.
 * @param layer The neural network layer position.
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
