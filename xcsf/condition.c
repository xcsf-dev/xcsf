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
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"       
#include "utils.h"
#include "condition.h"
#include "cond_dummy.h"
#include "cond_rectangle.h"
#include "cond_ellipsoid.h"
#include "cond_gp.h"
#include "cond_dgp.h"
#include "cond_neural.h"       
#include "cond_ternary.h"       
#include "action.h"
#include "rule_dgp.h"
#include "rule_neural.h"

/**
 * @brief Sets a classifier's condition functions to the implementations.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to set.
 */
void condition_set(const XCSF *xcsf, CL *c)
{
    switch(xcsf->COND_TYPE) {
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
