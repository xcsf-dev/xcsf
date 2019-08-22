/*
 * Copyright (C) 2012--2019 Richard Preen <rpreen@gmail.com>
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
 
#include <stdlib.h>
#include <stdio.h>
#include "neural_activations.h"

void activation_set(activate_ptr *activate, int func)
{
    switch(func) {
        case LOGISTIC: *activate = &logistic_activate; break;
        case RELU: *activate = &relu_activate; break;
        case GAUSSIAN: *activate = &gaussian_activate; break;
        case BENT_IDENTITY: *activate = &bent_identity_activate; break;
        case TANH: *activate = &tanh_activate; break;
        case SIN: *activate = &sin_activate; break;
        case COS: *activate = &cos_activate; break;
        case SOFT_PLUS: *activate = &soft_plus_activate; break;
        case IDENTITY: *activate = &identity_activate; break;
        case HARDTAN: *activate = &hardtan_activate; break;
        case STAIR: *activate = &stair_activate; break;
        case LEAKY: *activate = &leaky_activate; break;
        case ELU: *activate = &elu_activate; break;
        case RAMP: *activate = &ramp_activate; break;
        default:
            printf("activation_set(): invalid activation function: %d\n", func);
            exit(EXIT_FAILURE);
    }
}

void gradient_set(gradient_ptr *gradient, int func)
{
    switch(func) {
        case LOGISTIC: *gradient = &logistic_gradient; break;
        case RELU: *gradient = &relu_gradient; break;
        case GAUSSIAN: *gradient = &gaussian_gradient; break;
        case BENT_IDENTITY: *gradient = &bent_identity_gradient; break;
        case TANH: *gradient = &tanh_gradient; break;
        case SIN: *gradient = &sin_gradient; break;
        case COS: *gradient = &cos_gradient; break;
        case SOFT_PLUS: *gradient = &soft_plus_gradient; break;
        case IDENTITY: *gradient = &identity_gradient; break;
        case HARDTAN: *gradient = &hardtan_gradient; break;
        case STAIR: *gradient = &stair_gradient; break;
        case LEAKY: *gradient = &leaky_gradient; break;
        case ELU: *gradient = &elu_gradient; break;
        case RAMP: *gradient = &ramp_gradient; break;
        default:
            printf("gradient_set(): invalid activation function: %d\n", func);
            exit(EXIT_FAILURE);
    }
}

char *activation_string(int func)
{
     switch(func) {
        case LOGISTIC: return "logistic";
        case RELU: return "relu";
        case GAUSSIAN: return "gaussian";
        case BENT_IDENTITY: return "bent_identity";
        case TANH: return "tanh"; 
        case SIN: return "sin";
        case COS: return "cos";
        case SOFT_PLUS: return "soft_plus";
        case IDENTITY: return "identity";
        case HARDTAN: return "hardtan";
        case STAIR: return "stair";
        case LEAKY: return "leaky";
        case ELU: return "elu";
        case RAMP: return "ramp";
        default:
            printf("activation_string(): invalid activation function: %d\n", func);
            exit(EXIT_FAILURE);
    }
}
