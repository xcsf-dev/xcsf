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
 * @date 2015--2021.
 * @brief Interface for classifier actions.
 */

#pragma once

#include "xcsf.h"

#define ACT_TYPE_INTEGER (0) //!< Action type integer
#define ACT_TYPE_NEURAL (1) //!< Action type neural network

#define ACT_STRING_INTEGER ("integer\0") //!< Integer
#define ACT_STRING_NEURAL ("neural\0") //!< Neural

/**
 * @brief Parameters for initialising and operating actions.
 */
struct ArgsAct {
    int type; //!< Classifier action type
    struct ArgsLayer *largs; //!< Linked-list of layer parameters
};

void
action_set(const struct XCSF *xcsf, struct Cl *c);

const char *
action_type_as_string(const int type);

int
action_type_as_int(const char *type);

void
action_param_defaults(struct XCSF *xcsf);

void
action_param_free(struct XCSF *xcsf);

const char *
action_param_json_export(const struct XCSF *xcsf);

size_t
action_param_save(const struct XCSF *xcsf, FILE *fp);

size_t
action_param_load(struct XCSF *xcsf, FILE *fp);

/**
 * @brief Action interface data structure.
 * @details Action implementations must implement these functions.
 */
struct ActVtbl {
    bool (*act_impl_general)(const struct XCSF *xcsf, const struct Cl *c1,
                             const struct Cl *c2);
    bool (*act_impl_crossover)(const struct XCSF *xcsf, const struct Cl *c1,
                               const struct Cl *c2);
    bool (*act_impl_mutate)(const struct XCSF *xcsf, const struct Cl *c);
    int (*act_impl_compute)(const struct XCSF *xcsf, const struct Cl *c,
                            const double *x);
    void (*act_impl_copy)(const struct XCSF *xcsf, struct Cl *dest,
                          const struct Cl *src);
    void (*act_impl_cover)(const struct XCSF *xcsf, const struct Cl *c,
                           const double *x, const int action);
    void (*act_impl_free)(const struct XCSF *xcsf, const struct Cl *c);
    void (*act_impl_init)(const struct XCSF *xcsf, struct Cl *c);
    void (*act_impl_print)(const struct XCSF *xcsf, const struct Cl *c);
    void (*act_impl_update)(const struct XCSF *xcsf, const struct Cl *c,
                            const double *x, const double *y);
    size_t (*act_impl_save)(const struct XCSF *xcsf, const struct Cl *c,
                            FILE *fp);
    size_t (*act_impl_load)(const struct XCSF *xcsf, struct Cl *c, FILE *fp);
    const char *(*act_impl_json_export)(const struct XCSF *xcsf,
                                        const struct Cl *c);
};

/**
 * @brief Writes the action to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
static inline size_t
act_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    return (*c->act_vptr->act_impl_save)(xcsf, c, fp);
}

/**
 * @brief Reads the action from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
static inline size_t
act_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    return (*c->act_vptr->act_impl_load)(xcsf, c, fp);
}

/**
 * @brief Returns whether the action of classifier c1 is more general than c2.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The classifier whose action is tested to be more general.
 * @param [in] c2 The classifier whose action is tested to be more specific.
 * @return Whether the action of c1 is more general than c2.
 */
static inline bool
act_general(const struct XCSF *xcsf, const struct Cl *c1, const struct Cl *c2)
{
    return (*c1->act_vptr->act_impl_general)(xcsf, c1, c2);
}

/**
 * @brief Performs classifier action crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose action is being crossed.
 * @param [in] c2 The second classifier whose action is being crossed.
 * @return Whether any alterations were made.
 */
static inline bool
act_crossover(const struct XCSF *xcsf, const struct Cl *c1, const struct Cl *c2)
{
    return (*c1->act_vptr->act_impl_crossover)(xcsf, c1, c2);
}

/**
 * @brief Performs classifier action mutation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is being mutated.
 * @return Whether any alterations were made.
 */
static inline bool
act_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    return (*c->act_vptr->act_impl_mutate)(xcsf, c);
}

/**
 * @brief Computes the current classifier action using the input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier calculating the action.
 * @param [in] x The input state.
 * @return The classifier's action.
 */
static inline int
act_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    return (*c->act_vptr->act_impl_compute)(xcsf, c, x);
}

/**
 * @brief Copies the action from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
static inline void
act_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (*src->act_vptr->act_impl_copy)(xcsf, dest, src);
}

/**
 * @brief Generates an action that matches the specified value.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is being covered.
 * @param [in] x The input state to cover.
 * @param [in] action The action to cover.
 */
static inline void
act_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x,
          const int action)
{
    (*c->act_vptr->act_impl_cover)(xcsf, c, x, action);
}

/**
 * @brief Frees the memory used by the classifier action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be freed.
 */
static inline void
act_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (*c->act_vptr->act_impl_free)(xcsf, c);
}

/**
 * @brief Initialises a classifier's action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be initialised.
 */
static inline void
act_init(const struct XCSF *xcsf, struct Cl *c)
{
    (*c->act_vptr->act_impl_init)(xcsf, c);
}

/**
 * @brief Prints the classifier action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose action is to be printed.
 */
static inline void
act_print(const struct XCSF *xcsf, const struct Cl *c)
{
    (*c->act_vptr->act_impl_print)(xcsf, c);
}

/**
 * @brief Updates the classifier's action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose action is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
static inline void
act_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
           const double *y)
{
    (*c->act_vptr->act_impl_update)(xcsf, c, x, y);
}

/**
 * @brief Returns a json formatted string representation of an action .
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose action is to be returned.
 * @return String encoded in json format.
 */
static inline const char *
act_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    return (*c->act_vptr->act_impl_json_export)(xcsf, c);
}

/* parameter setters */

void
action_param_set_type_string(struct XCSF *xcsf, const char *a);

void
action_param_set_type(struct XCSF *xcsf, const int a);
