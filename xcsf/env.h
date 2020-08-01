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
 * @date 2015--2020.
 * @brief Built-in problem environment interface.
 */

#pragma once

void env_init(XCSF *xcsf, char **argv);

/**
 * @brief Built-in problem environment interface data structure.
 * @details Environment implementations must implement these functions.
 */
struct EnvVtbl {
    _Bool (*env_impl_isreset)(const XCSF *xcsf);
    _Bool (*env_impl_multistep)(const XCSF *xcsf);
    double (*env_impl_execute)(const XCSF *xcsf, int action);
    double (*env_impl_max_payoff)(const XCSF *xcsf);
    const double *(*env_impl_get_state)(const XCSF *xcsf);
    void (*env_impl_free)(const XCSF *xcsf);
    void (*env_impl_reset)(const XCSF *xcsf);
};

/**
 * @brief Returns whether the environment needs to be reset.
 * @param xcsf The XCSF data structure.
 * @return Whether the environment needs to be reset.
 */
static inline _Bool
env_is_reset(const XCSF *xcsf)
{
    return (*xcsf->env_vptr->env_impl_isreset)(xcsf);
}

/**
 * @brief Returns whether the environment is a multistep problem.
 * @param xcsf The XCSF data structure.
 * @return Whether the environment is multistep.
 */
static inline _Bool
env_multistep(const XCSF *xcsf)
{
    return (*xcsf->env_vptr->env_impl_multistep)(xcsf);
}

/**
 * @brief Executes the specified action and returns the payoff.
 * @param xcsf The XCSF data structure.
 * @param action The action to perform.
 * @return The payoff from performing the action.
 */
static inline double
env_execute(const XCSF *xcsf, int action)
{
    return (*xcsf->env_vptr->env_impl_execute)(xcsf, action);
}

/**
 * @brief Returns the maximum payoff value possible in the environment.
 * @param xcsf The XCSF data structure.
 * @return The maximum payoff.
 */
static inline double
env_max_payoff(const XCSF *xcsf)
{
    return (*xcsf->env_vptr->env_impl_max_payoff)(xcsf);
}

/**
 * @brief Returns the current environment perceptions.
 * @param xcsf The XCSF data structure.
 * @return The current perceptions.
 */
static inline const double *
env_get_state(const XCSF *xcsf)
{
    return (*xcsf->env_vptr->env_impl_get_state)(xcsf);
}

/**
 * @brief Frees the environment.
 * @param xcsf The XCSF data structure.
 */
static inline void
env_free(const XCSF *xcsf)
{
    (*xcsf->env_vptr->env_impl_free)(xcsf);
}

/**
 * @brief Resets the environment.
 * @param xcsf The XCSF data structure.
 */
static inline void
env_reset(const XCSF *xcsf)
{
    (*xcsf->env_vptr->env_impl_reset)(xcsf);
}
