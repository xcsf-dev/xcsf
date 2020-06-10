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
 * @file prediction.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Interface for classifier predictions.
 */

#pragma once

void prediction_set(const XCSF *xcsf, CL *c);

/**
 * @brief Prediction interface data structure.
 * @details Prediction implementations must implement these functions.
 */
struct PredVtbl {
    _Bool (*pred_impl_crossover)(const XCSF *xcsf, const CL *c1, const CL *c2);
    _Bool (*pred_impl_mutate)(const XCSF *xcsf, const CL *c);
    void (*pred_impl_compute)(const XCSF *xcsf, const CL *c, const double *x);
    void (*pred_impl_copy)(const XCSF *xcsf, CL *dest, const CL *src);
    void (*pred_impl_free)(const XCSF *xcsf, const CL *c);
    void (*pred_impl_init)(const XCSF *xcsf, CL *c);
    void (*pred_impl_print)(const XCSF *xcsf, const CL *c);
    void (*pred_impl_update)(const XCSF *xcsf, const CL *c, const double *x, const double *y);
    int (*pred_impl_size)(const XCSF *xcsf, const CL *c);
    size_t (*pred_impl_save)(const XCSF *xcsf, const CL *c, FILE *fp);
    size_t (*pred_impl_load)(const XCSF *xcsf, CL *c, FILE *fp);
};

/**
 * @brief Writes the prediction to a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is to be written.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
static inline size_t pred_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    return (*c->pred_vptr->pred_impl_save)(xcsf, c, fp);
}

/**
 * @brief Reads the prediction from a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is to be read.
 * @param fp Pointer to the file to be read.
 * @return The number of elements read.
 */
static inline size_t pred_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    return (*c->pred_vptr->pred_impl_load)(xcsf, c, fp);
}

/**
 * @brief Returns the size of the classifier prediction.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction size to return.
 * @return The size of the prediction.
 */
static inline int pred_size(const XCSF *xcsf, const CL *c)
{
    return (*c->pred_vptr->pred_impl_size)(xcsf, c);
}

/**
 * @brief Performs classifier prediction crossover.
 * @param xcsf The XCSF data structure.
 * @param c1 The first classifier whose prediction is being crossed.
 * @param c2 The second classifier whose prediction is being crossed.
 * @return Whether any alterations were made.
 */
static inline _Bool pred_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    return (*c1->pred_vptr->pred_impl_crossover)(xcsf, c1, c2);
}

/**
 * @brief Performs classifier prediction mutation.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is being mutated.
 * @return Whether any alterations were made.
 */
static inline _Bool pred_mutate(const XCSF *xcsf, const CL *c)
{
    return (*c->pred_vptr->pred_impl_mutate)(xcsf, c);
}

/**
 * @brief Computes the current classifier prediction using the input.
 * @param xcsf The XCSF data structure.
 * @param c The classifier calculating the prediction.
 * @param x The input state.
 */
static inline void pred_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    (*c->pred_vptr->pred_impl_compute)(xcsf, c, x);
}

/**
 * @brief Copies the prediction from one classifier to another.
 * @param xcsf The XCSF data structure.
 * @param dest The destination classifier.
 * @param src The source classifier.
 */
static inline void pred_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    (*src->pred_vptr->pred_impl_copy)(xcsf, dest, src);
}

/**
 * @brief Frees the memory used by the classifier prediction.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is to be freed.
 */
static inline void pred_free(const XCSF *xcsf, const CL *c)
{
    (*c->pred_vptr->pred_impl_free)(xcsf, c);
}

/**
 * @brief Initialises a classifier's prediction.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is to be initialised.
 */
static inline void pred_init(const XCSF *xcsf, CL *c)
{
    (*c->pred_vptr->pred_impl_init)(xcsf, c);
}

/**
 * @brief Prints the classifier prediction.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is to be printed.
 */
static inline void pred_print(const XCSF *xcsf, const CL *c)
{
    (*c->pred_vptr->pred_impl_print)(xcsf, c);
}

/**
 * @brief Updates the classifier's prediction.
 * @details Assumes the prediction has been computed for the current state.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose prediction is to be updated.
 * @param x The input state.
 * @param y The payoff value.
 */
static inline void pred_update(const XCSF *xcsf, const CL *c, const double *x,
                               const double *y)
{
    (*c->pred_vptr->pred_impl_update)(xcsf, c, x, y);
}

/**
 * @brief Prepares the input state for least squares computation.
 * @param xcsf The XCSF data structure.
 * @param x The input state.
 * @param tmp_input The transformed input (set by this function).
 */
static inline void pred_transform_input(const XCSF *xcsf, const double *x,
                                        double *tmp_input)
{
    // bias term
    tmp_input[0] = xcsf->PRED_X0;
    int idx = 1;
    // linear terms
    for(int i = 0; i < xcsf->x_dim; i++) {
        tmp_input[idx++] = x[i];
    }
    // quadratic terms
    if(xcsf->PRED_TYPE == PRED_TYPE_NLMS_QUADRATIC
            || xcsf->PRED_TYPE == PRED_TYPE_RLS_QUADRATIC) {
        for(int i = 0; i < xcsf->x_dim; i++) {
            for(int j = i; j < xcsf->x_dim; j++) {
                tmp_input[idx++] = x[i] * x[j];
            }
        }
    }
}
