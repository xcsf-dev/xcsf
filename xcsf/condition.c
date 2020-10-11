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
 * @file condition.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Interface for classifier conditions.
 */

#include "cond_dgp.h"
#include "cond_dummy.h"
#include "cond_ellipsoid.h"
#include "cond_gp.h"
#include "cond_neural.h"
#include "cond_rectangle.h"
#include "cond_ternary.h"
#include "rule_dgp.h"
#include "rule_neural.h"

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

/**
 * @brief Sets a classifier's condition functions to the implementations.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] c The classifier to set.
 */
void
condition_set(const struct XCSF *xcsf, struct Cl *c)
{
    switch (xcsf->COND_TYPE) {
        case COND_TYPE_DUMMY:
            c->cond_vptr = &cond_dummy_vtbl;
            break;
        case COND_TYPE_HYPERRECTANGLE:
            c->cond_vptr = &cond_rectangle_vtbl;
            break;
        case COND_TYPE_HYPERELLIPSOID:
            c->cond_vptr = &cond_ellipsoid_vtbl;
            break;
        case COND_TYPE_NEURAL:
            c->cond_vptr = &cond_neural_vtbl;
            break;
        case COND_TYPE_GP:
            c->cond_vptr = &cond_gp_vtbl;
            break;
        case COND_TYPE_DGP:
            c->cond_vptr = &cond_dgp_vtbl;
            break;
        case COND_TYPE_TERNARY:
            c->cond_vptr = &cond_ternary_vtbl;
            break;
        case RULE_TYPE_DGP:
            c->cond_vptr = &rule_dgp_cond_vtbl;
            c->act_vptr = &rule_dgp_act_vtbl;
            break;
        case RULE_TYPE_NEURAL:
            c->cond_vptr = &rule_neural_cond_vtbl;
            c->act_vptr = &rule_neural_act_vtbl;
            break;
        default:
            printf("Invalid condition type specified: %d\n", xcsf->COND_TYPE);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns a string representation of a condition type from an integer.
 * @param [in] type Integer representation of a condition type.
 * @return String representing the name of the condition type.
 */
const char *
condition_type_as_string(const int type)
{
    switch (type) {
        case COND_TYPE_DUMMY:
            return COND_STRING_DUMMY;
        case COND_TYPE_HYPERRECTANGLE:
            return COND_STRING_HYPERRECTANGLE;
        case COND_TYPE_HYPERELLIPSOID:
            return COND_STRING_HYPERELLIPSOID;
        case COND_TYPE_NEURAL:
            return COND_STRING_NEURAL;
        case COND_TYPE_GP:
            return COND_STRING_GP;
        case COND_TYPE_DGP:
            return COND_STRING_DGP;
        case COND_TYPE_TERNARY:
            return COND_STRING_TERNARY;
        case RULE_TYPE_DGP:
            return COND_STRING_RULE_DGP;
        case RULE_TYPE_NEURAL:
            return COND_STRING_RULE_NEURAL;
        case RULE_TYPE_NETWORK:
            return COND_STRING_RULE_NETWORK;
        default:
            printf("condition_type_as_string(): invalid type: %d\n", type);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Returns the integer representation of a condition type given a name.
 * @param [in] type String representation of a condition type.
 * @return Integer representing the condition type.
 */
int
condition_type_as_int(const char *type)
{
    if (strncmp(type, COND_STRING_DUMMY, 5) == 0) {
        return COND_TYPE_DUMMY;
    }
    if (strncmp(type, COND_STRING_HYPERRECTANGLE, 14) == 0) {
        return COND_TYPE_HYPERRECTANGLE;
    }
    if (strncmp(type, COND_STRING_HYPERELLIPSOID, 14) == 0) {
        return COND_TYPE_HYPERRECTANGLE;
    }
    if (strncmp(type, COND_STRING_NEURAL, 6) == 0) {
        return COND_TYPE_NEURAL;
    }
    if (strncmp(type, COND_STRING_GP, 7) == 0) {
        return COND_TYPE_GP;
    }
    if (strncmp(type, COND_STRING_DGP, 3) == 0) {
        return COND_TYPE_DGP;
    }
    if (strncmp(type, COND_STRING_TERNARY, 7) == 0) {
        return COND_TYPE_TERNARY;
    }
    if (strncmp(type, COND_STRING_RULE_DGP, 8) == 0) {
        return RULE_TYPE_DGP;
    }
    if (strncmp(type, COND_STRING_RULE_NEURAL, 11) == 0) {
        return RULE_TYPE_NEURAL;
    }
    if (strncmp(type, COND_STRING_RULE_NETWORK, 12) == 0) {
        return RULE_TYPE_NETWORK;
    }
    printf("condition_type_as_int(): invalid type: %s\n", type);
    exit(EXIT_FAILURE);
}
