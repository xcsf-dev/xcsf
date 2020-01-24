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
 * @file condition.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Interface for classifier conditions.
 */ 

#pragma once

void condition_set(XCSF *xcsf, CL *c);

/**
 * @brief Condition interface data structure.
 * @details Condition implementations must implement these functions.
 */ 
struct CondVtbl {
    /**
     * @brief Performs classifier condition crossover.
     * @param xcsf The XCSF data structure.
     * @param c1 The first classifier whose condition is being crossed.
     * @param c2 The second classifier whose condition is being crossed.
     * @return Whether any alterations were made.
     */
    _Bool (*cond_impl_crossover)(XCSF *xcsf, CL *c1, CL *c2);
    /**
     * @brief Returns whether the condition of classifier c1 is more general than c2.
     * @param xcsf The XCSF data structure.
     * @param c1 The classifier whose condition is tested to be more general.
     * @param c2 The classifier whose condition is tested to be more specific.
     * @return Whether the condition of c1 is more general than c2.
     */
    _Bool (*cond_impl_general)(XCSF *xcsf, CL *c1, CL *c2);
    /**
     * @brief Calculates whether the condition matches the input. 
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition to match.
     * @param x The input state.
     * @return Whether the condition matches the input.
     */
    _Bool (*cond_impl_match)(XCSF *xcsf, CL *c, const double *x);
    /**
     * @brief Performs classifier condition mutation.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is being mutated.
     * @return Whether any alterations were made.
     */
    _Bool (*cond_impl_mutate)(XCSF *xcsf, CL *c);
    /**
     * @brief Copies the condition from one classifier to another.
     * @param xcsf The XCSF data structure.
     * @param to The destination classifier.
     * @param from The source classifier.
     */
    void (*cond_impl_copy)(XCSF *xcsf, CL *to, CL *from);
    /**
     * @brief Generates a condition that matches the current input.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is being covered.
     * @param x The input state to cover.
     */
    void (*cond_impl_cover)(XCSF *xcsf, CL *c, const double *x);
    /**
     * @brief Frees the memory used by the classifier condition.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is to be freed.
     */
    void (*cond_impl_free)(XCSF *xcsf, CL *c);
    /**
     * @brief Initialises a classifier's condition.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is to be initialised.
     */
    void (*cond_impl_init)(XCSF *xcsf, CL *c);
    /**
     * @brief Prints the classifier condition.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is to be printed.
     */
    void (*cond_impl_print)(XCSF *xcsf, CL *c);
    /**
     * @brief Updates the classifier's condition.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is to be updated.
     * @param x The input state.
     * @param y The payoff value.
     */
    void (*cond_impl_update)(XCSF *xcsf, CL *c, const double *x, const double *y);
    /**
     * @brief Returns the size of the classifier condition.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition size to return.
     * @return The size of the condition.
     */
    int (*cond_impl_size)(XCSF *xcsf, CL *c);
    /**
     * @brief Writes the condition to a binary file.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is to be written.
     * @param fp Pointer to the file to be written.
     * @return The number of elements written.
     */
    size_t (*cond_impl_save)(XCSF *xcsf, CL *c, FILE *fp);
    /**
     * @brief Reads the condition from a binary file.
     * @param xcsf The XCSF data structure.
     * @param c The classifier whose condition is to be read.
     * @param fp Pointer to the file to be read.
     * @return The number of elements read.
     */
    size_t (*cond_impl_load)(XCSF *xcsf, CL *c, FILE *fp);
};

static inline size_t cond_save(XCSF *xcsf, CL *c, FILE *fp) {
    return (*c->cond_vptr->cond_impl_save)(xcsf, c, fp);
}

static inline size_t cond_load(XCSF *xcsf, CL *c, FILE *fp) {
    return (*c->cond_vptr->cond_impl_load)(xcsf, c, fp);
}

static inline int cond_size(XCSF *xcsf, CL *c) {
    return (*c->cond_vptr->cond_impl_size)(xcsf, c);
}

static inline void cond_update(XCSF *xcsf, CL *c, const double *x, const double *y) {
    (*c->cond_vptr->cond_impl_update)(xcsf, c, x, y);
}

static inline _Bool cond_crossover(XCSF *xcsf, CL *c1, CL *c2) {
    return (*c1->cond_vptr->cond_impl_crossover)(xcsf, c1, c2);
}

static inline _Bool cond_general(XCSF *xcsf, CL *c1, CL *c2) {
    return (*c1->cond_vptr->cond_impl_general)(xcsf, c1, c2);
}

static inline _Bool cond_match(XCSF *xcsf, CL *c, const double *x) {
    return (*c->cond_vptr->cond_impl_match)(xcsf, c, x);
}

static inline _Bool cond_mutate(XCSF *xcsf, CL *c) {
    return (*c->cond_vptr->cond_impl_mutate)(xcsf, c);
}

static inline void cond_copy(XCSF *xcsf, CL *to, CL *from) {
    (*from->cond_vptr->cond_impl_copy)(xcsf, to, from);
}

static inline void cond_cover(XCSF *xcsf, CL *c, const double *x) {
    (*c->cond_vptr->cond_impl_cover)(xcsf, c, x);
}

static inline void cond_free(XCSF *xcsf, CL *c) {
    (*c->cond_vptr->cond_impl_free)(xcsf, c);
}

static inline void cond_init(XCSF *xcsf, CL *c) {
    (*c->cond_vptr->cond_impl_init)(xcsf, c);
}

static inline void cond_print(XCSF *xcsf, CL *c) {
    (*c->cond_vptr->cond_impl_print)(xcsf, c);
}
