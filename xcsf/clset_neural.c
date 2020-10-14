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
 * @file clset_neural.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2020.
 * @brief Functions operating on sets of neural classifiers.
 */

#include "clset_neural.h"
#include "cl.h"
#include "cond_neural.h"
#include "condition.h"
#include "pred_neural.h"
#include "prediction.h"
#include "utils.h"

/**
 * @brief Calculates the mean number of condition layers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean.
 * @return The mean number of layers.
 */
double
clset_mean_cond_layers(const struct XCSF *xcsf, const struct Set *set)
{
    int sum = 0;
    int cnt = 0;
    if (xcsf->cond->type == COND_TYPE_NEURAL ||
        xcsf->cond->type == RULE_TYPE_NEURAL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            sum += cond_neural_layers(xcsf, iter->cl);
            ++cnt;
            iter = iter->next;
        }
    }
    if (cnt != 0) {
        return (double) sum / cnt;
    }
    return 0;
}
/**
 * @brief Calculates the mean number of condition neurons for a given layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean.
 * @param [in] layer The neural network layer position.
 * @return The mean number of neurons in the layer.
 */
double
clset_mean_cond_neurons(const struct XCSF *xcsf, const struct Set *set,
                        const int layer)
{
    int sum = 0;
    int cnt = 0;
    if (xcsf->cond->type == COND_TYPE_NEURAL ||
        xcsf->cond->type == RULE_TYPE_NEURAL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            sum += cond_neural_neurons(xcsf, iter->cl, layer);
            ++cnt;
            iter = iter->next;
        }
    }
    if (cnt != 0) {
        return (double) sum / cnt;
    }
    return 0;
}

/**
 * @brief Calculates the mean number of condition connections in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean.
 * @param [in] layer The position of layer to calculate.
 * @return The mean number of connections in the layer.
 */
double
clset_mean_cond_connections(const struct XCSF *xcsf, const struct Set *set,
                            const int layer)
{
    int sum = 0;
    int cnt = 0;
    if (xcsf->cond->type == COND_TYPE_NEURAL ||
        xcsf->cond->type == RULE_TYPE_NEURAL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            sum += cond_neural_connections(xcsf, iter->cl, layer);
            ++cnt;
            iter = iter->next;
        }
    }
    if (cnt != 0) {
        return (double) sum / cnt;
    }
    return 0;
}

/**
 * @brief Calculates the mean number of prediction neurons for a given layer.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean.
 * @param [in] layer The neural network layer position.
 * @return The mean number of neurons in the layer.
 */
double
clset_mean_pred_neurons(const struct XCSF *xcsf, const struct Set *set,
                        const int layer)
{
    int sum = 0;
    int cnt = 0;
    if (xcsf->pred->type == PRED_TYPE_NEURAL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            sum += pred_neural_neurons(xcsf, iter->cl, layer);
            ++cnt;
            iter = iter->next;
        }
    }
    if (cnt != 0) {
        return (double) sum / cnt;
    }
    return 0;
}

/**
 * @brief Calculates the mean number of prediction layers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean.
 * @return The mean number of layers.
 */
double
clset_mean_pred_layers(const struct XCSF *xcsf, const struct Set *set)
{
    int sum = 0;
    int cnt = 0;
    if (xcsf->pred->type == PRED_TYPE_NEURAL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            sum += pred_neural_layers(xcsf, iter->cl);
            ++cnt;
            iter = iter->next;
        }
    }
    if (cnt != 0) {
        return (double) sum / cnt;
    }
    return 0;
}

/**
 * @brief Calculates the mean prediction layer ETA of classifiers in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean.
 * @param [in] layer The position of layer to calculate.
 * @return The mean prediction layer ETA of classifiers in the set.
 */
double
clset_mean_pred_eta(const struct XCSF *xcsf, const struct Set *set,
                    const int layer)
{
    double sum = 0;
    int cnt = 0;
    if (xcsf->pred->type == PRED_TYPE_NEURAL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            sum += pred_neural_eta(xcsf, iter->cl, layer);
            ++cnt;
            iter = iter->next;
        }
    }
    if (cnt != 0) {
        return sum / cnt;
    }
    return 0;
}

/**
 * @brief Calculates the mean number of prediction connections in the set.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] set The set to calculate the mean.
 * @param [in] layer The position of layer to calculate.
 * @return The mean number of connections in the layer.
 */
double
clset_mean_pred_connections(const struct XCSF *xcsf, const struct Set *set,
                            const int layer)
{
    int sum = 0;
    int cnt = 0;
    if (xcsf->pred->type == PRED_TYPE_NEURAL) {
        const struct Clist *iter = set->list;
        while (iter != NULL) {
            sum += pred_neural_connections(xcsf, iter->cl, layer);
            ++cnt;
            iter = iter->next;
        }
    }
    if (cnt != 0) {
        return (double) sum / cnt;
    }
    return 0;
}
