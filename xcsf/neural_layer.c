/*
 * Copyright (C) 2015--2019 Richard Preen <rpreen@gmail.com>
 *
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xcsf.h"       
#include "neural_activations.h"
#include "neural_layer.h"
#include "neural_layer_connected.h"

void neural_layer_init(LAYER *l, int type, int num_inputs, int num_outputs, int activation)
{
    switch(type) {
        case 0:
            l->layer_vptr = &layer_connected_vtbl;
            break;
        default:
            printf("neural_layer_init(): invalid layer type: %d\n", type);
            exit(EXIT_FAILURE);
    }
    layer_init(l, num_inputs, num_outputs, activation);
}
