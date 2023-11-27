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
 * @date 2015--2023.
 * @brief Prediction array functions.
 */

#include "pa.h"
#include "cl.h"
#include "utils.h"

/**
 * @brief Resets the prediction array to zero.
 * @param [in] xcsf The XCSF data structure.
 */
static void
pa_reset(const struct XCSF *xcsf)
{
    for (int i = 0; i < xcsf->pa_size; ++i) {
        xcsf->pa[i] = 0;
        xcsf->nr[i] = 0;
    }
}

/**
 * @brief Initialises the prediction array.
 * @param [in] xcsf The XCSF data structure.
 */
void
pa_init(struct XCSF *xcsf)
{
    xcsf->pa_size = xcsf->n_actions * xcsf->y_dim;
    xcsf->pa = malloc(sizeof(double) * xcsf->pa_size);
    xcsf->nr = malloc(sizeof(double) * xcsf->pa_size);
    xcsf->cover = calloc(xcsf->pa_size, sizeof(double));
}

/**
 * @brief Builds the prediction array for the specified input.
 * @details Calculates the match set mean fitness weighted prediction for each
 * action. For supervised learning n_actions=1; reinforcement learning y_dim=1.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] x The input state.
 */
void
pa_build(const struct XCSF *xcsf, const double *x)
{
    const struct Set *set = &xcsf->mset;
    double *pa = xcsf->pa;
    double *nr = xcsf->nr;
    pa_reset(xcsf);
#ifdef PARALLEL_PRED
    // (parallel) propagate input and compute predictions
    struct Cl *clist[set->size];
    struct Clist *iter = set->list;
    for (int i = 0; i < set->size; ++i) {
        clist[i] = NULL;
        if (iter != NULL) {
            clist[i] = iter->cl;
            iter = iter->next;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < set->size; ++i) {
        if (clist[i] != NULL) {
            cl_predict(xcsf, clist[i], x);
        }
    }
#else
    // (series) propagate input and compute predictions
    const struct Clist *iter = set->list;
    while (iter != NULL) {
        cl_predict(xcsf, iter->cl, x);
        iter = iter->next;
    }
#endif
    // compute the prediction array in series for determinism
    iter = set->list;
    while (iter != NULL) {
        const double *pred = iter->cl->prediction;
        const double fitness = iter->cl->fit;
        for (int j = 0; j < xcsf->y_dim; ++j) {
            pa[iter->cl->action * xcsf->y_dim + j] += pred[j] * fitness;
            nr[iter->cl->action * xcsf->y_dim + j] += fitness;
        }
        iter = iter->next;
    }
    for (int i = 0; i < xcsf->n_actions; ++i) {
        for (int j = 0; j < xcsf->y_dim; ++j) {
            const int k = i * xcsf->y_dim + j;
            if (nr[k] != 0) {
                pa[k] /= nr[k];
            } else {
                pa[k] = 0;
            }
        }
    }
}

/**
 * @brief Returns the best action in the prediction array.
 * @details Ties broken uniformly random.
 * @param [in] xcsf The XCSF data structure.
 * @return The best action.
 */
int
pa_best_action(const struct XCSF *xcsf)
{
    int *max_i = calloc(xcsf->n_actions, sizeof(int));
    double max = xcsf->pa[0];
    int n_max = 1;
    for (int i = 1; i < xcsf->n_actions; ++i) {
        const double val = xcsf->pa[i];
        if (val > max) {
            max = val;
            max_i[0] = i;
            n_max = 1;
        } else if (val == max) {
            max_i[n_max] = i;
            ++n_max;
        }
    }
    const int best = max_i[rand_uniform_int(0, n_max)];
    free(max_i);
    return best;
}

/**
 * @brief Returns a random action from the prediction array.
 * @param [in] xcsf The XCSF data structure.
 * @return A random action.
 */
int
pa_rand_action(const struct XCSF *xcsf)
{
    int action = 0;
    do {
        action = rand_uniform_int(0, xcsf->n_actions);
    } while (xcsf->nr[action] == 0);
    return action;
}

/**
 * @brief Returns the highest value in the prediction array.
 * @param [in] xcsf The XCSF data structure.
 * @return The highest value in the prediction array.
 */
double
pa_best_val(const struct XCSF *xcsf)
{
    const int max_i = argmax(xcsf->pa, xcsf->pa_size);
    return xcsf->pa[max_i];
}

/**
 * @brief Returns the value of a specified action in the prediction array.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] action The specified action.
 * @return The value of the action in the prediction array.
 */
double
pa_val(const struct XCSF *xcsf, const int action)
{
    if (action < 0 || action >= xcsf->n_actions) {
        printf("pa_val() error: invalid action specified: %d\n", action);
        exit(EXIT_FAILURE);
    }
    return xcsf->pa[action];
}

/**
 * @brief Frees the prediction array.
 * @param [in] xcsf The XCSF data structure.
 */
void
pa_free(const struct XCSF *xcsf)
{
    free(xcsf->pa);
    free(xcsf->nr);
    free(xcsf->cover);
}
