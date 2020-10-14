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
 * @file env_mux.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief The real multiplexer problem environment.
 *
 * @details Generates random real vectors of length k+pow(2,k) where the
 * first k bits determine the position of the output bit in the last pow(2,k)
 * bits. E.g., for a 3-bit problem, the first rounded bit addresses which of
 * the following 2 bits are the (rounded) output.
 *
 * Example valid lengths: 3, 6, 11, 20, 37, 70, 135, 264.
 */

#include "env_mux.h"
#include "param.h"
#include "utils.h"

#define MAX_PAYOFF (1.) //!< Payoff provided for making a correct classification

/**
 * @brief Initialises a real multiplexer environment of specified length.
 * @details The biggest mux problem is chosen that fits the specified length.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] bits The problem length.
 */
void
env_mux_init(struct XCSF *xcsf, const int bits)
{
    struct EnvMux *env = malloc(sizeof(struct EnvMux));
    env->pos_bits = 1;
    while (env->pos_bits + pow(2, env->pos_bits) <= bits) {
        ++(env->pos_bits);
    }
    --(env->pos_bits);
    const int n = env->pos_bits + (int) pow(2, env->pos_bits);
    env->state = malloc(sizeof(double) * n);
    xcsf->env = env;
    param_init(xcsf, n, 1, 2);
}

/**
 * @brief Frees the multiplexer environment.
 * @param [in] xcsf The XCSF data structure.
 */
void
env_mux_free(const struct XCSF *xcsf)
{
    struct EnvMux *env = xcsf->env;
    free(env->state);
    free(env);
}

/**
 * @brief Returns a random multiplexer problem instance.
 * @param [in] xcsf The XCSF data structure.
 * @return A random multiplexer problem.
 */
const double *
env_mux_get_state(const struct XCSF *xcsf)
{
    const struct EnvMux *env = xcsf->env;
    for (int i = 0; i < xcsf->x_dim; ++i) {
        env->state[i] = rand_uniform(0, 1);
    }
    return env->state;
}

/**
 * @brief Returns the reward for executing a multiplexer action.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] action The selected action.
 * @return The payoff from performing the action.
 */
double
env_mux_execute(const struct XCSF *xcsf, const int action)
{
    const struct EnvMux *env = xcsf->env;
    int pos = env->pos_bits;
    for (int i = 0; i < env->pos_bits; ++i) {
        if (env->state[i] > 0.5) {
            pos += (int) pow(2, (double) (env->pos_bits - 1 - i));
        }
    }
    const int answer = (env->state[pos] > 0.5) ? 1 : 0;
    return (action == answer) ? MAX_PAYOFF : 0;
}

/**
 * @brief Dummy method since no multiplexer reset is necessary.
 * @param [in] xcsf The XCSF data structure.
 */
void
env_mux_reset(const struct XCSF *xcsf)
{
    (void) xcsf;
}

/**
 * @brief Returns whether the multiplexer is in a terminal state.
 * @param [in] xcsf The XCSF data structure.
 * @return True.
 */
bool
env_mux_is_done(const struct XCSF *xcsf)
{
    (void) xcsf;
    return true;
}

/**
 * @brief Returns the maximum payoff value possible in the multiplexer.
 * @param [in] xcsf The XCSF data structure.
 * @return The maximum payoff.
 */
double
env_mux_maxpayoff(const struct XCSF *xcsf)
{
    (void) xcsf;
    return MAX_PAYOFF;
}

/**
 * @brief Returns whether the multiplexer is a multistep problem.
 * @param [in] xcsf The XCSF data structure.
 * @return False.
 */
bool
env_mux_multistep(const struct XCSF *xcsf)
{
    (void) xcsf;
    return false;
}
