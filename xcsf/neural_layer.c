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
 * @file neural_layer.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2016--2019.
 * @brief Interface for neural network layers.
 */ 
      
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "xcsf.h"
#include "utils.h"
#include "neural_activations.h"
#include "neural.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"
#include "neural_layer_dropout.h"
#include "neural_layer_noise.h"
#include "neural_layer_softmax.h"

/**
 * @brief Sets a neural network layer's functions to the implementations.
 * @param xcsf The XCSF data structure.
 * @param l The neural network layer to set.
 */
void neural_layer_set_vptr(LAYER *l)
{
    switch(l->layer_type) {
        case CONNECTED: 
            l->layer_vptr = &layer_connected_vtbl;
            break;
        case DROPOUT:
            l->layer_vptr = &layer_dropout_vtbl;
            break;
        case NOISE:
            l->layer_vptr = &layer_noise_vtbl;
            break;
        case SOFTMAX:
            l->layer_vptr = &layer_softmax_vtbl;
            break;
        default:
            printf("Error setting layer vptr for type: %d\n", l->layer_type);
            exit(EXIT_FAILURE);
    }   
}
