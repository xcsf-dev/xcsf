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
 * @date 2015--2019.
 * @brief Interface for classifier predictions.
 */ 
  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"       
#include "utils.h"
#include "prediction.h"
#include "pred_nlms.h"
#include "pred_rls.h"
#include "pred_neural.h"     

void prediction_set(XCSF *xcsf, CL *c)
{
    switch(xcsf->PRED_TYPE) {
        case 0:
        case 1:
            c->pred_vptr = &pred_nlms_vtbl;
            break;
        case 2:
        case 3:
            c->pred_vptr = &pred_rls_vtbl;
            break;
        case 4:
            c->pred_vptr = &pred_neural_vtbl;
            break;
        default:
            printf("Invalid prediction type specified: %d\n", xcsf->PRED_TYPE);
            exit(EXIT_FAILURE);
    }
}
