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
 * @file env_mux.h
 * @brief The real multiplexer problem environment.
 * @details Generates random real vectors of length k+pow(2,k) where the
 * first k bits determine the position of the output bit in the last pow(2,k)
 * bits. E.g., for a 3-bit problem, the first rounded bit addresses which of
 * the following 2 bits are the output.
 */ 

#pragma once

_Bool env_mux_isreset(XCSF *xcsf);
_Bool env_mux_multistep(XCSF *xcsf);
double env_mux_execute(XCSF *xcsf, int action);
double env_mux_maxpayoff(XCSF *xcsf);
double *env_mux_get_state(XCSF *xcsf);
void env_mux_free(XCSF *xcsf);
void env_mux_init(XCSF *xcsf, int bits);
void env_mux_reset(XCSF *xcsf);

static struct EnvVtbl const env_mux_vtbl = {
    &env_mux_isreset,
    &env_mux_multistep,
    &env_mux_execute,
    &env_mux_maxpayoff,
    &env_mux_get_state,
    &env_mux_free,
    &env_mux_reset
};      
