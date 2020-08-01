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
 * @file action.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Interface for classifier actions.
 */

#pragma once

void action_set(const XCSF *xcsf, CL *c);

/**
 * @brief Action interface data structure.
 * @details Action implementations must implement these functions.
 */
struct ActVtbl {
    _Bool (*act_impl_general)(const XCSF *xcsf, const CL *c1, const CL *c2);
    _Bool (*act_impl_crossover)(const XCSF *xcsf, const CL *c1, const CL *c2);
    _Bool (*act_impl_mutate)(const XCSF *xcsf, const CL *c);
    int (*act_impl_compute)(const XCSF *xcsf, const CL *c, const double *x);
    void (*act_impl_copy)(const XCSF *xcsf, CL *dest, const CL *src);
    void (*act_impl_cover)(const XCSF *xcsf, const CL *c, const double *x,
                           int action);
    void (*act_impl_free)(const XCSF *xcsf, const CL *c);
    void (*act_impl_init)(const XCSF *xcsf, CL *c);
    void (*act_impl_print)(const XCSF *xcsf, const CL *c);
    void (*act_impl_update)(const XCSF *xcsf, const CL *c, const double *x,
                            const double *y);
    size_t (*act_impl_save)(const XCSF *xcsf, const CL *c, FILE *fp);
    size_t (*act_impl_load)(const XCSF *xcsf, CL *c, FILE *fp);
};

/**
 * @brief Writes the action to a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be written.
 * @param fp Pointer to the file to be written.
 * @return The number of elements written.
 */
static inline size_t
act_save(const XCSF *xcsf, const CL *c, FILE *fp)
{
    return (*c->act_vptr->act_impl_save)(xcsf, c, fp);
}

/**
 * @brief Reads the action from a binary file.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be read.
 * @param fp Pointer to the file to be read.
 * @return The number of elements read.
 */
static inline size_t
act_load(const XCSF *xcsf, CL *c, FILE *fp)
{
    return (*c->act_vptr->act_impl_load)(xcsf, c, fp);
}

/**
 * @brief Returns whether the action of classifier c1 is more general than c2.
 * @param xcsf The XCSF data structure.
 * @param c1 The classifier whose action is tested to be more general.
 * @param c2 The classifier whose action is tested to be more specific.
 * @return Whether the action of c1 is more general than c2.
 */
static inline _Bool
act_general(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    return (*c1->act_vptr->act_impl_general)(xcsf, c1, c2);
}

/**
 * @brief Performs classifier action crossover.
 * @param xcsf The XCSF data structure.
 * @param c1 The first classifier whose action is being crossed.
 * @param c2 The second classifier whose action is being crossed.
 * @return Whether any alterations were made.
 */
static inline _Bool
act_crossover(const XCSF *xcsf, const CL *c1, const CL *c2)
{
    return (*c1->act_vptr->act_impl_crossover)(xcsf, c1, c2);
}

/**
 * @brief Performs classifier action mutation.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is being mutated.
 * @return Whether any alterations were made.
 */
static inline _Bool
act_mutate(const XCSF *xcsf, const CL *c)
{
    return (*c->act_vptr->act_impl_mutate)(xcsf, c);
}

/**
 * @brief Computes the current classifier action using the input.
 * @param xcsf The XCSF data structure.
 * @param c The classifier calculating the action.
 * @param x The input state.
 * @return The classifier's action.
 */
static inline int
act_compute(const XCSF *xcsf, const CL *c, const double *x)
{
    return (*c->act_vptr->act_impl_compute)(xcsf, c, x);
}

/**
 * @brief Copies the action from one classifier to another.
 * @param xcsf The XCSF data structure.
 * @param dest The destination classifier.
 * @param src The source classifier.
 */
static inline void
act_copy(const XCSF *xcsf, CL *dest, const CL *src)
{
    (*src->act_vptr->act_impl_copy)(xcsf, dest, src);
}

/**
 * @brief Generates an action that matches the specified value.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is being covered.
 * @param x The input state to cover.
 * @param action The action to cover.
 */
static inline void
act_cover(const XCSF *xcsf, const CL *c, const double *x, int action)
{
    (*c->act_vptr->act_impl_cover)(xcsf, c, x, action);
}

/**
 * @brief Frees the memory used by the classifier action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be freed.
 */
static inline void
act_free(const XCSF *xcsf, const CL *c)
{
    (*c->act_vptr->act_impl_free)(xcsf, c);
}

/**
 * @brief Initialises a classifier's action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be initialised.
 */
static inline void
act_init(const XCSF *xcsf, CL *c)
{
    (*c->act_vptr->act_impl_init)(xcsf, c);
}

/**
 * @brief Prints the classifier action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be printed.
 */
static inline void
act_print(const XCSF *xcsf, const CL *c)
{
    (*c->act_vptr->act_impl_print)(xcsf, c);
}

/**
 * @brief Updates the classifier's action.
 * @param xcsf The XCSF data structure.
 * @param c The classifier whose action is to be updated.
 * @param x The input state.
 * @param y The payoff value.
 */
static inline void
act_update(const XCSF *xcsf, const CL *c, const double *x,
           const double *y)
{
    (*c->act_vptr->act_impl_update)(xcsf, c, x, y);
}
