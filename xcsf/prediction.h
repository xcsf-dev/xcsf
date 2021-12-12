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
 * @date 2015--2021.
 * @brief Interface for classifier predictions.
 */

#pragma once

#include "xcsf.h"

#define PRED_TYPE_CONSTANT (0) //!< Prediction type constant
#define PRED_TYPE_NLMS_LINEAR (1) //!< Prediction type linear nlms
#define PRED_TYPE_NLMS_QUADRATIC (2) //!< Prediction type quadratic nlms
#define PRED_TYPE_RLS_LINEAR (3) //!< Prediction type linear rls
#define PRED_TYPE_RLS_QUADRATIC (4) //!< Prediction type quadratic rls
#define PRED_TYPE_NEURAL (5) //!< Prediction type neural

#define PRED_STRING_CONSTANT ("constant\0") //!< Constant
#define PRED_STRING_NLMS_LINEAR ("nlms_linear\0") //!< Linear nlms
#define PRED_STRING_NLMS_QUADRATIC ("nlms_quadratic\0") //!< Quadratic nlms
#define PRED_STRING_RLS_LINEAR ("rls_linear\0") //!< Linear rls
#define PRED_STRING_RLS_QUADRATIC ("rls_quadratic\0") //!< Quadratic rls
#define PRED_STRING_NEURAL ("neural\0") //!< Neural

/**
 * @brief Parameters for initialising and operating predictions.
 */
struct ArgsPred {
    int type; //!< Classifier prediction type: least squares, etc.
    bool evolve_eta; //!< Whether to evolve the gradient descent rate
    double eta; //!< Gradient descent rate
    double eta_min; //!< Minimum gradient descent rate
    double lambda; //!< RLS forget rate
    double scale_factor; //!< Initial values for the RLS gain-matrix
    double x0; //!< Prediction weight vector offset value
    struct ArgsLayer *largs; //!< Linked-list of layer parameters
};

const char *
prediction_type_as_string(const int type);

int
prediction_type_as_int(const char *type);

size_t
pred_param_load(struct XCSF *xcsf, FILE *fp);

size_t
pred_param_save(const struct XCSF *xcsf, FILE *fp);

void
pred_param_defaults(struct XCSF *xcsf);

void
pred_param_free(struct XCSF *xcsf);

const char *
pred_param_json_export(const struct XCSF *xcsf);

void
pred_transform_input(const struct XCSF *xcsf, const double *x, const double X0,
                     double *tmp_input);

void
prediction_set(const struct XCSF *xcsf, struct Cl *c);

/**
 * @brief Prediction interface data structure.
 * @details Prediction implementations must implement these functions.
 */
struct PredVtbl {
    bool (*pred_impl_crossover)(const struct XCSF *xcsf, const struct Cl *c1,
                                const struct Cl *c2);
    bool (*pred_impl_mutate)(const struct XCSF *xcsf, const struct Cl *c);
    void (*pred_impl_compute)(const struct XCSF *xcsf, const struct Cl *c,
                              const double *x);
    void (*pred_impl_copy)(const struct XCSF *xcsf, struct Cl *dest,
                           const struct Cl *src);
    void (*pred_impl_free)(const struct XCSF *xcsf, const struct Cl *c);
    void (*pred_impl_init)(const struct XCSF *xcsf, struct Cl *c);
    void (*pred_impl_print)(const struct XCSF *xcsf, const struct Cl *c);
    void (*pred_impl_update)(const struct XCSF *xcsf, const struct Cl *c,
                             const double *x, const double *y);
    double (*pred_impl_size)(const struct XCSF *xcsf, const struct Cl *c);
    size_t (*pred_impl_save)(const struct XCSF *xcsf, const struct Cl *c,
                             FILE *fp);
    size_t (*pred_impl_load)(const struct XCSF *xcsf, struct Cl *c, FILE *fp);
    const char *(*pred_impl_json_export)(const struct XCSF *xcsf,
                                         const struct Cl *c);
};

/**
 * @brief Writes the prediction to a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return The number of elements written.
 */
static inline size_t
pred_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    return (*c->pred_vptr->pred_impl_save)(xcsf, c, fp);
}

/**
 * @brief Reads the prediction from a file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be read.
 * @param [in] fp Pointer to the file to be read.
 * @return The number of elements read.
 */
static inline size_t
pred_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    return (*c->pred_vptr->pred_impl_load)(xcsf, c, fp);
}

/**
 * @brief Returns the size of the classifier prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction size to return.
 * @return The size of the prediction.
 */
static inline double
pred_size(const struct XCSF *xcsf, const struct Cl *c)
{
    return (*c->pred_vptr->pred_impl_size)(xcsf, c);
}

/**
 * @brief Performs classifier prediction crossover.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c1 The first classifier whose prediction is being crossed.
 * @param [in] c2 The second classifier whose prediction is being crossed.
 * @return Whether any alterations were made.
 */
static inline bool
pred_crossover(const struct XCSF *xcsf, const struct Cl *c1,
               const struct Cl *c2)
{
    return (*c1->pred_vptr->pred_impl_crossover)(xcsf, c1, c2);
}

/**
 * @brief Performs classifier prediction mutation.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is being mutated.
 * @return Whether any alterations were made.
 */
static inline bool
pred_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    return (*c->pred_vptr->pred_impl_mutate)(xcsf, c);
}

/**
 * @brief Computes the current classifier prediction using the input.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier calculating the prediction.
 * @param [in] x The input state.
 */
static inline void
pred_compute(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    (*c->pred_vptr->pred_impl_compute)(xcsf, c, x);
}

/**
 * @brief Copies the prediction from one classifier to another.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] dest The destination classifier.
 * @param [in] src The source classifier.
 */
static inline void
pred_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (*src->pred_vptr->pred_impl_copy)(xcsf, dest, src);
}

/**
 * @brief Frees the memory used by the classifier prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be freed.
 */
static inline void
pred_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (*c->pred_vptr->pred_impl_free)(xcsf, c);
}

/**
 * @brief Initialises a classifier's prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be initialised.
 */
static inline void
pred_init(const struct XCSF *xcsf, struct Cl *c)
{
    (*c->pred_vptr->pred_impl_init)(xcsf, c);
}

/**
 * @brief Prints the classifier prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier whose prediction is to be printed.
 */
static inline void
pred_print(const struct XCSF *xcsf, const struct Cl *c)
{
    (*c->pred_vptr->pred_impl_print)(xcsf, c);
}

/**
 * @brief Updates the classifier's prediction.
 * @pre The prediction has been computed for the current state.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose prediction is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
static inline void
pred_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
            const double *y)
{
    (*c->pred_vptr->pred_impl_update)(xcsf, c, x, y);
}

/**
 * @brief Returns a json formatted string representation of a prediction.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c Classifier whose prediction is to be returned.
 * @return String encoded in json format.
 */
static inline const char *
pred_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    return (*c->pred_vptr->pred_impl_json_export)(xcsf, c);
}

/* parameter setters */

void
pred_param_set_eta(struct XCSF *xcsf, const double a);

void
pred_param_set_eta_min(struct XCSF *xcsf, const double a);

void
pred_param_set_lambda(struct XCSF *xcsf, const double a);

void
pred_param_set_scale_factor(struct XCSF *xcsf, const double a);

void
pred_param_set_x0(struct XCSF *xcsf, const double a);

void
pred_param_set_evolve_eta(struct XCSF *xcsf, const bool a);

void
pred_param_set_type(struct XCSF *xcsf, const int a);

void
pred_param_set_type_string(struct XCSF *xcsf, const char *a);
