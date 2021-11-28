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
 * @date 2015--2021.
 * @brief Functions operating on sets of classifiers.
 */

#include "clset.h"
#include "cl.h"
#include "utils.h"

#define MAX_COVER (1000000) //!< Maximum number of covering attempts

/**
 * @brief Finds a rule in the population that never matches an input.
 * @param [in] xcsf The XCSF data structure.
 * @param [out] del A pointer to the classifier to be deleted, if one is found.
 * @param [out] delprev A pointer to the rule previous to the one being deleted.
 */
static void
clset_pset_never_match(const struct XCSF *xcsf, struct Clist **del,
                       struct Clist **delprev)
{
    struct Clist *prev = NULL;
    struct Clist *iter = xcsf->pset.list;
    while (iter != NULL) {
        if (iter->cl->mtotal == 0 && iter->cl->age > xcsf->M_PROBATION) {
            *del = iter;
            *delprev = prev;
            break;
        }
        prev = iter;
        iter = iter->next;
    }
}

/**
 * @brief Selects a classifier from the population for deletion via roulette.
 * @details If compaction is enabled and the average system error is below E0,
 * two classifiers are selected using roulette wheel selection with the
 * deletion vote and the rule with the largest condition + prediction size is
 * chosen. For fixed-length representations, the effect is the same as one
 * roulete spin.
 * @param [in] xcsf The XCSF data structure.
 * @param [out] del A pointer to the rule to be deleted.
 * @param [out] delprev A pointer to the rule previous to the one being deleted.
 */
static void
clset_pset_roulette(const struct XCSF *xcsf, struct Clist **del,
                    struct Clist **delprev)
{
    const double avg_fit = clset_total_fit(&xcsf->pset) / xcsf->pset.num;
    double total_vote = 0;
    struct Clist *iter = xcsf->pset.list;
    while (iter != NULL) {
        total_vote += cl_del_vote(xcsf, iter->cl, avg_fit);
        iter = iter->next;
    }
    double delsize = 0;
    const int n_spins = (xcsf->COMPACTION && xcsf->error < xcsf->E0) ? 2 : 1;
    for (int i = 0; i < n_spins; ++i) {
        // perform a single roulette spin with the deletion vote
        iter = xcsf->pset.list;
        struct Clist *prev = NULL;
        const double p = rand_uniform(0, total_vote);
        double sum = cl_del_vote(xcsf, iter->cl, avg_fit);
        while (p > sum) {
            prev = iter;
            iter = iter->next;
            sum += cl_del_vote(xcsf, iter->cl, avg_fit);
        }
        // select the rule for deletion if it is the largest sized winner
        const double s =
            cl_cond_size(xcsf, iter->cl) + cl_pred_size(xcsf, iter->cl);
        if (*del == NULL || s > delsize) {
            *del = iter;
            *delprev = prev;
            delsize = s;
        }
    }
}

/**
 * @brief Deletes a single classifier from the population set.
 * @param [in] xcsf The XCSF data structure.
 */
static void
clset_pset_del(struct XCSF *xcsf)
{
    struct Clist *del = NULL;
    struct Clist *delprev = NULL;
    // select any rules that never match
    clset_pset_never_match(xcsf, &del, &delprev);
    // if none found, select a rule using roulette wheel
    if (del == NULL) {
        clset_pset_roulette(xcsf, &del, &delprev);
    }
    // decrement numerosity
    --(del->cl->num);
    --(xcsf->pset.num);
    // remove macro-classifiers as necessary
    if (del->cl->num == 0) {
        clset_add(&xcsf->kset, del->cl);
        --(xcsf->pset.size);
        if (delprev == NULL) {
            xcsf->pset.list = del->next;
        } else {
            delprev->next = del->next;
        }
        free(del);
    }
}

/**
 * @brief Checks whether each action is covered by the match set.
 * @param [in] xcsf The XCSF data structure.
 * @param [out] act_covered Vector specifying whether each action is covered.
 * @return Whether all actions are covered.
 */
static bool
clset_action_coverage(const struct XCSF *xcsf, bool *act_covered)
{
    memset(act_covered, 0, sizeof(bool) * xcsf->n_actions);
    const struct Clist *iter = xcsf->mset.list;
    while (iter != NULL) {
        act_covered[iter->cl->action] = true;
        iter = iter->next;
    }
    for (int i = 0; i < xcsf->n_actions; ++i) {
        if (!act_covered[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Ensures all possible actions are covered by the match set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] x The input state.
 */
static void
clset_cover(struct XCSF *xcsf, const double *x)
{
    int attempts = 0;
    bool *act_covered = malloc(sizeof(bool) * xcsf->n_actions);
    bool covered = clset_action_coverage(xcsf, act_covered);
    while (!covered) {
        covered = true;
        for (int i = 0; i < xcsf->n_actions; ++i) {
            if (!act_covered[i]) {
                // create a new classifier with matching condition and action
                struct Cl *new = malloc(sizeof(struct Cl));
                cl_init(xcsf, new, (xcsf->mset.num) + 1, xcsf->time);
                cl_cover(xcsf, new, x, i);
                clset_add(&xcsf->pset, new);
                clset_add(&xcsf->mset, new);
            }
        }
        // enforce population size
        const int prev_psize = xcsf->pset.size;
        clset_pset_enforce_limit(xcsf);
        // if a macro classifier was deleted,
        // remove any deleted rules from the match set
        if (prev_psize > xcsf->pset.size) {
            const int prev_msize = xcsf->mset.size;
            clset_validate(&xcsf->mset);
            // if the deleted classifier was in the match set,
            // check if an action is now not covered
            if (prev_msize > xcsf->mset.size) {
                covered = clset_action_coverage(xcsf, act_covered);
            }
        }
        ++attempts;
        if (attempts > MAX_COVER) {
            printf("Error: max covering attempts (%d) exceeded\n", MAX_COVER);
            exit(EXIT_FAILURE);
        }
    }
    free(act_covered);
}

/**
 * @brief Updates the fitness of classifiers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to update.
 */
static void
clset_update_fit(const struct XCSF *xcsf, const struct Set *set)
{
    double acc_sum = 0;
    double accs[set->size];
    // calculate accuracies
    const struct Clist *iter = set->list;
    for (int i = 0; iter != NULL && i < set->size; ++i) {
        accs[i] = cl_acc(xcsf, iter->cl);
        acc_sum += accs[i] * iter->cl->num;
        iter = iter->next;
    }
    // update fitnesses
    iter = set->list;
    for (int i = 0; iter != NULL && i < set->size; ++i) {
        cl_update_fit(xcsf, iter->cl, acc_sum, accs[i]);
        iter = iter->next;
    }
}

/**
 * @brief Performs set subsumption.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to perform subsumption.
 */
static void
clset_subsumption(struct XCSF *xcsf, struct Set *set)
{
    // find the most general subsumer in the set
    struct Cl *s = NULL;
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        struct Cl *c = iter->cl;
        if (cl_subsumer(xcsf, c) && (s == NULL || cl_general(xcsf, c, s))) {
            s = c;
        }
        iter = iter->next;
    }
    // subsume the more specific classifiers in the set
    if (s != NULL) {
        bool subsumed = false;
        iter = set->list;
        while (iter != NULL) {
            struct Cl *c = iter->cl;
            if (c != NULL && s != c && cl_general(xcsf, s, c)) {
                s->num += c->num;
                c->num = 0;
                clset_add(&xcsf->kset, c);
                subsumed = true;
            }
            iter = iter->next;
        }
        if (subsumed) {
            clset_validate(set);
            clset_validate(&xcsf->pset);
        }
    }
}

/**
 * @brief Calculates the total time stamps of classifiers in the set.
 * @param [in] set The set to calculate the total time.
 * @return The total time of classifiers in the set.
 */
static double
clset_total_time(const struct Set *set)
{
    double sum = 0;
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        sum += iter->cl->time * iter->cl->num;
        iter = iter->next;
    }
    return sum;
}

/**
 * @brief Initialises a new population of random classifiers.
 * @param [in] xcsf The XCSF data structure.
 */
void
clset_pset_init(struct XCSF *xcsf)
{
    if (xcsf->POP_INIT) {
        while (xcsf->pset.num < xcsf->POP_SIZE) {
            struct Cl *new = malloc(sizeof(struct Cl));
            cl_init(xcsf, new, xcsf->POP_SIZE, 0);
            cl_rand(xcsf, new);
            clset_add(&xcsf->pset, new);
        }
    }
}

/**
 * @brief Initialises a new set.
 * @param [in] set The set to be initialised.
 */
void
clset_init(struct Set *set)
{
    set->list = NULL;
    set->size = 0;
    set->num = 0;
}

/**
 * @brief Enforces the maximum population size limit.
 * @param [in] xcsf The XCSF data structure.
 */
void
clset_pset_enforce_limit(struct XCSF *xcsf)
{
    while (xcsf->pset.num > xcsf->POP_SIZE) {
        clset_pset_del(xcsf);
    }
}

/**
 * @brief Constructs the match set - forward propagates conditions and actions.
 * @details Processes the matching conditions and actions for each classifier
 * in the population. If a classifier matches, it is added to the match set.
 * Covering is performed if any actions are unrepresented.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] x The input state.
 */
void
clset_match(struct XCSF *xcsf, const double *x)
{
#ifdef PARALLEL_MATCH
    // prepare for parallel processing of matching conditions
    struct Clist *blist[xcsf->pset.size];
    struct Clist *iter = xcsf->pset.list;
    for (int i = 0; iter != NULL && i < xcsf->pset.size; ++i) {
        blist[i] = iter;
        iter = iter->next;
    }
    // process conditions and actions setting m flags in parallel
    #pragma omp parallel for
    for (int i = 0; i < xcsf->pset.size; ++i) {
        cl_match(xcsf, blist[i]->cl, x);
        cl_action(xcsf, blist[i]->cl, x);
    }
    // build match set list in series
    for (int i = 0; i < xcsf->pset.size; ++i) {
        if (cl_m(xcsf, blist[i]->cl)) {
            clset_add(&xcsf->mset, blist[i]->cl);
        }
    }
#else
    // process conditions and actions and build match set list in series
    struct Clist *iter = xcsf->pset.list;
    while (iter != NULL) {
        if (cl_match(xcsf, iter->cl, x)) {
            clset_add(&xcsf->mset, iter->cl);
            cl_action(xcsf, iter->cl, x);
        }
        iter = iter->next;
    }
#endif
    // perform covering if all actions are not represented
    if (xcsf->n_actions > 1 || xcsf->mset.size < 1) {
        clset_cover(xcsf, x);
    }
    // update statistics
    xcsf->mset_size += (xcsf->mset.size - xcsf->mset_size) * xcsf->BETA;
    xcsf->mfrac += (clset_mfrac(xcsf) - xcsf->mfrac) * xcsf->BETA;
}

/**
 * @brief Constructs the action set from the match set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] action The action used to build the set.
 */
void
clset_action(struct XCSF *xcsf, const int action)
{
    const struct Clist *iter = xcsf->mset.list;
    while (iter != NULL) {
        if (iter->cl->action == action) {
            clset_add(&xcsf->aset, iter->cl);
        }
        iter = iter->next;
    }
    // update statistics
    xcsf->aset_size += (xcsf->aset.size - xcsf->aset_size) * xcsf->BETA;
}

/**
 * @brief Adds a classifier to the set.
 * @param [in] set The set to add the classifier.
 * @param [in] c The classifier to add.
 */
void
clset_add(struct Set *set, struct Cl *c)
{
    if (set->list == NULL) {
        set->list = malloc(sizeof(struct Clist));
        set->list->cl = c;
        set->list->next = NULL;
    } else {
        struct Clist *new = malloc(sizeof(struct Clist));
        new->cl = c;
        new->next = set->list;
        set->list = new;
    }
    ++(set->size);
    set->num += c->num;
}

/**
 * @brief Provides reinforcement to the set and performs set subsumption.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to provide reinforcement.
 * @param [in] x The input state.
 * @param [in] y The payoff from the environment.
 * @param [in] cur Whether the update is for the current or previous state.
 */
void
clset_update(struct XCSF *xcsf, struct Set *set, const double *x,
             const double *y, const bool cur)
{
#ifdef PARALLEL_UPDATE
    struct Clist *blist[set->size];
    struct Clist *iter = set->list;
    for (int i = 0; iter != NULL && i < set->size; ++i) {
        blist[i] = iter;
        iter = iter->next;
    }
    #pragma omp parallel for
    for (int i = 0; i < set->size; ++i) {
        cl_update(xcsf, blist[i]->cl, x, y, set->num, cur);
    }
#else
    struct Clist *iter = set->list;
    while (iter != NULL) {
        cl_update(xcsf, iter->cl, x, y, set->num, cur);
        iter = iter->next;
    }
#endif
    clset_update_fit(xcsf, set);
    if (xcsf->SET_SUBSUMPTION) {
        clset_subsumption(xcsf, set);
    }
}

/**
 * @brief Removes classifiers with 0 numerosity from the set.
 * @param [in] set The set to validate.
 */
void
clset_validate(struct Set *set)
{
    set->size = 0;
    set->num = 0;
    struct Clist *prev = NULL;
    struct Clist *iter = set->list;
    while (iter != NULL) {
        if (iter->cl == NULL || iter->cl->num == 0) {
            if (prev == NULL) {
                set->list = iter->next;
                free(iter);
                iter = set->list;
            } else {
                prev->next = iter->next;
                free(iter);
                iter = prev->next;
            }
        } else {
            ++(set->size);
            set->num += iter->cl->num;
            prev = iter;
            iter = iter->next;
        }
    }
}

/**
 * @brief Prints the classifiers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to print.
 * @param [in] print_cond Whether to print the conditions.
 * @param [in] print_act Whether to print the actions.
 * @param [in] print_pred Whether to print the predictions.
 */
void
clset_print(const struct XCSF *xcsf, const struct Set *set,
            const bool print_cond, const bool print_act, const bool print_pred)
{
    printf("%s\n",
           clset_json_export(xcsf, set, print_cond, print_act, print_pred));
}

/**
 * @brief Sets the time stamps for classifiers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to update the time stamps.
 */
void
clset_set_times(const struct XCSF *xcsf, const struct Set *set)
{
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        iter->cl->time = xcsf->time;
        iter = iter->next;
    }
}

/**
 * @brief Calculates the total fitness of classifiers in the set.
 * @param [in] set The set to calculate the total fitness.
 * @return The total fitness of classifiers in the set.
 */
double
clset_total_fit(const struct Set *set)
{
    double sum = 0;
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        sum += iter->cl->fit;
        iter = iter->next;
    }
    return sum;
}

/**
 * @brief Calculates the mean time stamp of classifiers in the set.
 * @param [in] set The set to calculate the mean time.
 * @return The mean time of classifiers in the set.
 */
double
clset_mean_time(const struct Set *set)
{
    return clset_total_time(set) / set->num;
}

/**
 * @brief Frees the set, but not the classifiers.
 * @param [in] set The set to free.
 */
void
clset_free(struct Set *set)
{
    struct Clist *iter = set->list;
    while (iter != NULL) {
        set->list = iter->next;
        free(iter);
        iter = set->list;
    }
    set->size = 0;
    set->num = 0;
}

/**
 * @brief Frees the set and the classifiers.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to free.
 */
void
clset_kill(const struct XCSF *xcsf, struct Set *set)
{
    struct Clist *iter = set->list;
    while (iter != NULL) {
        cl_free(xcsf, iter->cl);
        set->list = iter->next;
        free(iter);
        iter = set->list;
    }
    set->size = 0;
    set->num = 0;
}

/**
 * @brief Writes the population set to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
size_t
clset_pset_save(const struct XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    s += fwrite(&xcsf->pset.size, sizeof(int), 1, fp);
    s += fwrite(&xcsf->pset.num, sizeof(int), 1, fp);
    const struct Clist *iter = xcsf->pset.list;
    while (iter != NULL) {
        s += cl_save(xcsf, iter->cl, fp);
        iter = iter->next;
    }
    return s;
}

/**
 * @brief Reads the population set from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
size_t
clset_pset_load(struct XCSF *xcsf, FILE *fp)
{
    size_t s = 0;
    int size = 0;
    int num = 0;
    s += fread(&size, sizeof(int), 1, fp);
    s += fread(&num, sizeof(int), 1, fp);
    clset_init(&xcsf->pset);
    for (int i = 0; i < size; ++i) {
        struct Cl *c = malloc(sizeof(struct Cl));
        s += cl_load(xcsf, c, fp);
        clset_add(&xcsf->pset, c);
    }
    return s;
}

/**
 * @brief Calculates the mean condition size of classifiers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean condition size.
 * @return The mean condition size of classifiers in the set.
 */
double
clset_mean_cond_size(const struct XCSF *xcsf, const struct Set *set)
{
    double sum = 0;
    int cnt = 0;
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        sum += cl_cond_size(xcsf, iter->cl);
        ++cnt;
        iter = iter->next;
    }
    return sum / cnt;
}

/**
 * @brief Calculates the mean prediction size of classifiers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean prediction size.
 * @return The mean prediction size of classifiers in the set.
 */
double
clset_mean_pred_size(const struct XCSF *xcsf, const struct Set *set)
{
    double sum = 0;
    int cnt = 0;
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        sum += cl_pred_size(xcsf, iter->cl);
        ++cnt;
        iter = iter->next;
    }
    return sum / cnt;
}

/**
 * @brief Returns the fraction of inputs matched by the most general rule with
 * error below E0. If no rules below E0, the lowest error rule is used.
 * @param [in] xcsf The XCSF data structure.
 * @return The fraction of inputs matched.
 */
double
clset_mfrac(const struct XCSF *xcsf)
{
    double mfrac = 0;
    // most general rule below E0
    const struct Clist *iter = xcsf->pset.list;
    while (iter != NULL) {
        const double e = iter->cl->err;
        if (e < xcsf->E0 && iter->cl->exp * xcsf->BETA > 1) {
            const double m = cl_mfrac(xcsf, iter->cl);
            if (m > mfrac) {
                mfrac = m;
            }
        }
        iter = iter->next;
    }
    // lowest error rule
    if (mfrac == 0) {
        double error = DBL_MAX;
        iter = xcsf->pset.list;
        while (iter != NULL) {
            const double e = iter->cl->err;
            if (e < error && iter->cl->exp * xcsf->BETA > 1) {
                mfrac = cl_mfrac(xcsf, iter->cl);
                error = e;
            }
            iter = iter->next;
        }
    }
    return mfrac;
}

/**
 * @brief Returns a json formatted string representation of a classifier set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to be returned.
 * @param [in] return_cond Whether to return the condition.
 * @param [in] return_act Whether to return the action.
 * @param [in] return_pred Whether to return the prediction.
 * @return String encoded in json format.
 */
const char *
clset_json_export(const struct XCSF *xcsf, const struct Set *set,
                  const bool return_cond, const bool return_act,
                  const bool return_pred)
{
    cJSON *json = cJSON_CreateObject();
    cJSON *classifiers = cJSON_AddArrayToObject(json, "classifiers");
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        const char *str = cl_json_export(xcsf, iter->cl, return_cond,
                                         return_act, return_pred);
        cJSON *classifier = cJSON_Parse(str);
        cJSON_AddItemToArray(classifiers, classifier);
        iter = iter->next;
    }
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
