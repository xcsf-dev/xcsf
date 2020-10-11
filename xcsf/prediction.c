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
 * @file prediction.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Interface for classifier predictions.
 */

#include "pred_constant.h"
#include "pred_neural.h"
#include "pred_nlms.h"
#include "pred_rls.h"

/**
 * @brief Sets a classifier's prediction functions to the implementations.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to set.
 */
void
prediction_set(const struct XCSF *xcsf, struct Cl *c)
{
    switch (xcsf->PRED_TYPE) {
        case PRED_TYPE_CONSTANT:
            c->pred_vptr = &pred_constant_vtbl;
            break;
        case PRED_TYPE_NLMS_LINEAR:
        case PRED_TYPE_NLMS_QUADRATIC:
            c->pred_vptr = &pred_nlms_vtbl;
            break;
        case PRED_TYPE_RLS_LINEAR:
        case PRED_TYPE_RLS_QUADRATIC:
            c->pred_vptr = &pred_rls_vtbl;
            break;
        case PRED_TYPE_NEURAL:
            c->pred_vptr = &pred_neural_vtbl;
            break;
        default:
            printf("prediction_set(): invalid type: %d\n", xcsf->PRED_TYPE);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns a string representation of a prediction type from the integer.
 * @param [in] type Integer representation of a prediction type.
 * @return String representing the name of the prediction type.
 */
const char *
prediction_type_as_string(const int type)
{
    switch (type) {
        case PRED_TYPE_CONSTANT:
            return PRED_STRING_CONSTANT;
        case PRED_TYPE_NLMS_LINEAR:
            return PRED_STRING_NLMS_LINEAR;
        case PRED_TYPE_NLMS_QUADRATIC:
            return PRED_STRING_NLMS_QUADRATIC;
        case PRED_TYPE_RLS_LINEAR:
            return PRED_STRING_RLS_LINEAR;
        case PRED_TYPE_RLS_QUADRATIC:
            return PRED_STRING_RLS_QUADRATIC;
        case PRED_TYPE_NEURAL:
            return PRED_STRING_NEURAL;
        default:
            printf("prediction_type_as_string(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the integer representation of a prediction type given a name.
 * @param [in] type String representation of a prediction type.
 * @return Integer representing the prediction type.
 */
int
prediction_type_as_int(const char *type)
{
    if (strncmp(type, PRED_STRING_CONSTANT, 9) == 0) {
        return PRED_TYPE_CONSTANT;
    }
    if (strncmp(type, PRED_STRING_NLMS_LINEAR, 12) == 0) {
        return PRED_TYPE_NLMS_LINEAR;
    }
    if (strncmp(type, PRED_STRING_NLMS_QUADRATIC, 15) == 0) {
        return PRED_TYPE_NLMS_QUADRATIC;
    }
    if (strncmp(type, PRED_STRING_RLS_LINEAR, 11) == 0) {
        return PRED_TYPE_RLS_LINEAR;
    }
    if (strncmp(type, PRED_STRING_RLS_QUADRATIC, 14) == 0) {
        return PRED_TYPE_RLS_QUADRATIC;
    }
    if (strncmp(type, PRED_STRING_NEURAL, 7) == 0) {
        return PRED_TYPE_NEURAL;
    }
    printf("prediction_type_as_int(): invalid type: %s\n", type);
    exit(EXIT_FAILURE);
}
