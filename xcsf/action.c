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
 * @file action.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Interface for classifier actions.
 */

#include "action.h"
#include "act_integer.h"
#include "utils.h"

/**
 * @brief Sets a classifier's action functions to the implementations.
 * @param xcsf The XCSF data structure.
 * @param c The classifier to set.
 */
void
action_set(const struct XCSF *xcsf, struct CL *c)
{
    switch (xcsf->ACT_TYPE) {
        case ACT_TYPE_INTEGER:
            c->act_vptr = &act_integer_vtbl;
            break;
        default:
            printf("Invalid action type specified: %d\n", xcsf->ACT_TYPE);
            exit(EXIT_FAILURE);
    }
}
