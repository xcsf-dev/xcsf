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
     
/**
 * @file env_mux.c
 * @brief The real multiplexer problem environment.
 * @details Generates random real vectors of length k+pow(2,k) where the
 * first k bits determine the position of the output bit in the last pow(2,k)
 * bits. E.g., for a 3-bit problem, the first rounded bit addresses which of
 * the following 2 bits are the output.
 */ 
   
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcsf.h"
#include "utils.h"
#include "env.h"
#include "env_mux.h"

/**
 * @brief Real multiplexer environment data structure.
 */  
typedef struct ENV_MUX {
    double *state; //!< current state
    int pos_bits; //!< number of position bits
} ENV_MUX;

#define MAX_PAYOFF 1000.0
char *mux_cstate(XCSF *xcsf);

void env_mux_init(XCSF *xcsf, int bits)
{
    ENV_MUX *env = malloc(sizeof(ENV_MUX));
    env->pos_bits = 1;
    while(env->pos_bits + pow(2, env->pos_bits) <= bits) {
        (env->pos_bits)++;
    }
    (env->pos_bits)--;
    env->state = malloc(sizeof(double) * bits);
    xcsf->num_actions = 2;
    xcsf->num_x_vars = bits;
    xcsf->num_y_vars = 1;
    xcsf->env = env;
}

void env_mux_free(XCSF *xcsf)
{
    ENV_MUX *env = xcsf->env;
    free(env->state);
    free(env);
}

double *env_mux_get_state(XCSF *xcsf)
{
    ENV_MUX *env = xcsf->env;
    for(int i = 0; i < xcsf->num_x_vars; i++) {
        env->state[i] = rand_uniform(0,1);
    }
    return env->state;
}

double env_mux_execute(XCSF *xcsf, int action)
{
    ENV_MUX *env = xcsf->env;
    int pos = env->pos_bits;
    for(int i = 0; i < env->pos_bits; i++) {
        if(env->state[i] > 0.5) {
            pos += pow(2, (double)(env->pos_bits - 1 - i));
        }
    }
    int answer = (env->state[pos] > 0.5) ? 1 : 0;
    if(action == answer) {
        return MAX_PAYOFF;
    }
    else {
        return 0;
    }
}  
  
void env_mux_reset(XCSF *xcsf)
{
    (void)xcsf;
}
 
_Bool env_mux_isreset(XCSF *xcsf)
{
    (void)xcsf;
    return true;
}
 
double env_mux_maxpayoff(XCSF *xcsf)
{
    (void)xcsf;
    return MAX_PAYOFF;
}

_Bool env_mux_multistep(XCSF *xcsf)
{
    (void)xcsf;
    return false;
}
