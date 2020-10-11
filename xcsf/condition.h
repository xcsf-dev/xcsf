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

#include "xcsf.h"

#define COND_TYPE_DUMMY (0) //!< Condition type dummy
#define COND_TYPE_HYPERRECTANGLE (1) //!< Condition type hyperrectangle
#define COND_TYPE_HYPERELLIPSOID (2) //!< Condition type hyperellipsoid
#define COND_TYPE_NEURAL (3) //!< Condition type neural network
#define COND_TYPE_GP (4) //!< Condition type tree GP
#define COND_TYPE_DGP (5) //!< Condition type DGP
#define COND_TYPE_TERNARY (6) //!< Condition type ternary
#define RULE_TYPE_DGP (11) //!< Condition type and action type DGP
#define RULE_TYPE_NEURAL (12) //!< Condition type and action type neural
#define RULE_TYPE_NETWORK (13) //!< Condition type and prediction type neural

#define COND_STRING_DUMMY ("dummy") //!< Dummy
#define COND_STRING_HYPERRECTANGLE ("hyperrectangle") //!< Hyperrectangle
#define COND_STRING_HYPERELLIPSOID ("hyperellipsoid") //!< Hyperellipsoid
#define COND_STRING_NEURAL ("neural") //!< Neural
#define COND_STRING_GP ("tree-gp") //!< Tree GP
#define COND_STRING_DGP ("dgp") //!< DGP
#define COND_STRING_TERNARY ("ternary") //!< Ternary
#define COND_STRING_RULE_DGP ("rule-dgp") //!< Rule DGP
#define COND_STRING_RULE_NEURAL ("rule-neural") //!< Rule neural
#define COND_STRING_RULE_NETWORK ("rule-network") //!< Rule network

void
condition_set(const struct XCSF *xcsf, struct Cl *c);

const char *
condition_type_as_string(const int type);

int
condition_type_as_int(const char *type);

/**
 * @brief Condition interface data structure.
 * @details Condition implementations must implement these functions.
 */
struct CondVtbl {
    bool (*cond_impl_crossover)(const struct XCSF *xcsf, const struct Cl *c1,
                                const struct Cl *c2);
    bool (*cond_impl_general)(const struct XCSF *xcsf, const struct Cl *c1,
                              const struct Cl *c2);
    bool (*cond_impl_match)(const struct XCSF *xcsf, const struct Cl *c,
                            const double *x);
    bool (*cond_impl_mutate)(const struct XCSF *xcsf, const struct Cl *c);
    void (*cond_impl_copy)(const struct XCSF *xcsf, struct Cl *dest,
                           const struct Cl *src);
    void (*cond_impl_cover)(const struct XCSF *xcsf, const struct Cl *c,
                            const double *x);
    void (*cond_impl_free)(const struct XCSF *xcsf, const struct Cl *c);
    void (*cond_impl_init)(const struct XCSF *xcsf, struct Cl *c);
    void (*cond_impl_print)(const struct XCSF *xcsf, const struct Cl *c);
    void (*cond_impl_update)(const struct XCSF *xcsf, const struct Cl *c,
                             const double *x, const double *y);
    double (*cond_impl_size)(const struct XCSF *xcsf, const struct Cl *c);
    size_t (*cond_impl_save)(const struct XCSF *xcsf, const struct Cl *c,
                             FILE *fp);
    size_t (*cond_impl_load)(const struct XCSF *xcsf, struct Cl *c, FILE *fp);
};

/**
 * @brief Writes the condition to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
static inline size_t
cond_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    return (*c->cond_vptr->cond_impl_save)(xcsf, c, fp);
}

/**
 * @brief Reads the condition from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
static inline size_t
cond_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    return (*c->cond_vptr->cond_impl_load)(xcsf, c, fp);
}

/**
 * @brief Returns the size of the classifier condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition size to return.
 * @return The size of the condition.
 */
static inline double
cond_size(const struct XCSF *xcsf, const struct Cl *c)
{
    return (*c->cond_vptr->cond_impl_size)(xcsf, c);
}

/**
 * @brief Updates the classifier's condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
static inline void
cond_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
            const double *y)
{
    (*c->cond_vptr->cond_impl_update)(xcsf, c, x, y);
}

/**
 * @brief Performs classifier condition crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose condition is being crossed.
 * @param [in] c2 The second classifier whose condition is being crossed.
 * @return Whether any alterations were made.
 */
static inline bool
cond_crossover(const struct XCSF *xcsf, const struct Cl *c1,
               const struct Cl *c2)
{
    return (*c1->cond_vptr->cond_impl_crossover)(xcsf, c1, c2);
}

/**
 * @brief Returns whether classifier c1 has a condition more general than c2.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The classifier whose condition is tested to be more general.
 * @param [in] c2 The classifier whose condition is tested to be more specific.
 * @return Whether the condition of c1 is more general than c2.
 */
static inline bool
cond_general(const struct XCSF *xcsf, const struct Cl *c1, const struct Cl *c2)
{
    return (*c1->cond_vptr->cond_impl_general)(xcsf, c1, c2);
}

/**
 * @brief Calculates whether the condition matches the input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition to match.
 * @param [in] x The input state.
 * @return Whether the condition matches the input.
 */
static inline bool
cond_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    return (*c->cond_vptr->cond_impl_match)(xcsf, c, x);
}

/**
 * @brief Performs classifier condition mutation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is being mutated.
 * @return Whether any alterations were made.
 */
static inline bool
cond_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    return (*c->cond_vptr->cond_impl_mutate)(xcsf, c);
}

/**
 * @brief Copies the condition from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
static inline void
cond_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (*src->cond_vptr->cond_impl_copy)(xcsf, dest, src);
}

/**
 * @brief Generates a condition that matches the current input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is being covered.
 * @param [in] x The input state to cover.
 */
static inline void
cond_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    (*c->cond_vptr->cond_impl_cover)(xcsf, c, x);
}

/**
 * @brief Frees the memory used by the classifier condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be freed.
 */
static inline void
cond_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (*c->cond_vptr->cond_impl_free)(xcsf, c);
}

/**
 * @brief Initialises a classifier's condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be initialised.
 */
static inline void
cond_init(const struct XCSF *xcsf, struct Cl *c)
{
    (*c->cond_vptr->cond_impl_init)(xcsf, c);
}

/**
 * @brief Prints the classifier condition.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose condition is to be printed.
 */
static inline void
cond_print(const struct XCSF *xcsf, const struct Cl *c)
{
    (*c->cond_vptr->cond_impl_print)(xcsf, c);
}
