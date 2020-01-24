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
    /**
     * @brief Performs classifier prediction crossover.
     * @param xcsf The XCSF data structure.
     * @param c1 The first classifier whose prediction is being crossed.
     * @param c2 The second classifier whose prediction is being crossed.
     * @return Whether any alterations were made.
     */
    _Bool (*pred_impl_crossover)(const XCSF *xcsf, CL *c1, CL *c2);
    /**
     * @brief Performs classifier prediction mutation.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction is being mutated.
     * @return Whether any alterations were made.
     */
    _Bool (*pred_impl_mutate)(const XCSF *xcsf, CL *c);
    /**
     * @brief Computes the current classifier prediction using the input.
     * @param xcsf The XCSF data structure.
     * @param c The classifier calculating the prediction.
     * @param x The input state.
     * @return The classifier's prediction.
     */
    const double *(*pred_impl_compute)(const XCSF *xcsf, CL *c, const double *x);
    /**
     * @brief Copies the prediction from one classifier to another.
     * @param xcsf The XCSF data structure.
     * @param to The destination classifier.
     * @param from The source classifier.
     */
    void (*pred_impl_copy)(const XCSF *xcsf, CL *to,  CL *from);
    /**
     * @brief Frees the memory used by the classifier prediction.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction is to be freed.
     */
    void (*pred_impl_free)(const XCSF *xcsf, CL *c);
    /**
     * @brief Initialises a classifier's prediction.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction is to be initialised.
     */
    void (*pred_impl_init)(const XCSF *xcsf, CL *c);
    /**
     * @brief Prints the classifier prediction.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction is to be printed.
     */
    void (*pred_impl_print)(const XCSF *xcsf, CL *c);
    /**
     * @brief Updates the classifier's prediction.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction is to be updated.
     * @param x The input state.
     * @param y The payoff value.
     */
    void (*pred_impl_update)(const XCSF *xcsf, CL *c, const double *x, const double *y);
    /**
     * @brief Returns the size of the classifier prediction.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction size to return.
     * @return The size of the prediction.
     */
    int (*pred_impl_size)(const XCSF *xcsf, CL *c);
    /**
     * @brief Writes the prediction to a binary file.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction is to be written.
     * @param fp Pointer to the file to be written.
     * @return The number of elements written.
     */
    size_t (*pred_impl_save)(const XCSF *xcsf, CL *c, FILE *fp);
    /**
     * @brief Reads the prediction from a binary file.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose prediction is to be read.
     * @param fp Pointer to the file to be read.
     * @return The number of elements read.
     */
    size_t (*pred_impl_load)(const XCSF *xcsf, CL *c, FILE *fp);
};

static inline size_t pred_save(const XCSF *xcsf, CL *c, FILE *fp) {
    return (*c->pred_vptr->pred_impl_save)(xcsf, c, fp);
}

static inline size_t pred_load(const XCSF *xcsf, CL *c, FILE *fp) {
    return (*c->pred_vptr->pred_impl_load)(xcsf, c, fp);
}

static inline int pred_size(const XCSF *xcsf, CL *c) {
    return (*c->pred_vptr->pred_impl_size)(xcsf, c);
}

static inline _Bool pred_crossover(const XCSF *xcsf, CL *c1, CL *c2) {
    return (*c1->pred_vptr->pred_impl_crossover)(xcsf, c1, c2);
}

static inline _Bool pred_mutate(const XCSF *xcsf, CL *c) {
    return (*c->pred_vptr->pred_impl_mutate)(xcsf, c);
}

static inline const double *pred_compute(const XCSF *xcsf, CL *c, const double *x) {
    return (*c->pred_vptr->pred_impl_compute)(xcsf, c, x);
}

static inline void pred_copy(const XCSF *xcsf, CL *to, CL *from) {
    (*from->pred_vptr->pred_impl_copy)(xcsf, to, from);
}

static inline void pred_free(const XCSF *xcsf, CL *c) {
    (*c->pred_vptr->pred_impl_free)(xcsf, c);
}

static inline void pred_init(const XCSF *xcsf, CL *c) {
    (*c->pred_vptr->pred_impl_init)(xcsf, c);
}

static inline void pred_print(const XCSF *xcsf, CL *c) {
    (*c->pred_vptr->pred_impl_print)(xcsf, c);
}

static inline void pred_update(const XCSF *xcsf, CL *c, const double *x, const double *y) {
    (*c->pred_vptr->pred_impl_update)(xcsf, c, x, y);
}
