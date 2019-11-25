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
 * @file env.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2019.
 * @brief Built-in problem environment interface.
 */ 

#pragma once

void env_init(XCSF *xcsf, char **argv);

/**
 * @brief Built-in problem environment interface data structure.
 * @details Environment implementations must implement these functions.
 */ 
struct EnvVtbl {
    _Bool (*env_impl_isreset)(XCSF *xcsf);
    _Bool (*env_impl_multistep)(XCSF *xcsf);
    double (*env_impl_execute)(XCSF *xcsf, int action);
    double (*env_impl_max_payoff)(XCSF *xcsf);
    double *(*env_impl_get_state)(XCSF *xcsf);
    void (*env_impl_free)(XCSF *xcsf);
    void (*env_impl_reset)(XCSF *xcsf);
};

static inline _Bool env_is_reset(XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_isreset)(xcsf);
}

static inline _Bool env_multistep(XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_multistep)(xcsf);
}

static inline double env_execute(XCSF *xcsf, int action) {
    return (*xcsf->env_vptr->env_impl_execute)(xcsf, action);
}

static inline double env_max_payoff(XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_max_payoff)(xcsf);
}

static inline double *env_get_state(XCSF *xcsf) {
    return (*xcsf->env_vptr->env_impl_get_state)(xcsf);
}

static inline void env_free(XCSF *xcsf) {
    (*xcsf->env_vptr->env_impl_free)(xcsf);
}

static inline void env_reset(XCSF *xcsf) {
    (*xcsf->env_vptr->env_impl_reset)(xcsf);
}
