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
 * @file xcs_rl.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Reinforcement learning functions.
 * @details A trial consists of one or more steps.
 */

#include "xcs_rl.h"
#include "clset.h"
#include "ea.h"
#include "env.h"
#include "pa.h"
#include "param.h"
#include "perf.h"
#include "utils.h"

/**
 * @brief Executes a reinforcement learning trial using a built-in environment.
 * @param [in] xcsf The XCSF data structure.
 * @param [out] error The mean system prediction error.
 * @param [in] explore Whether this is an exploration or exploitation trial.
 * @return Returns the accuracy for single-step problems and the number of
 * steps taken to reach the goal for multi-step problems.
 */
static double
xcs_rl_trial(struct XCSF *xcsf, double *error, const bool explore)
{
    env_reset(xcsf);
    param_set_explore(xcsf, explore);
    xcs_rl_init_trial(xcsf);
    *error = 0; // mean prediction error over all steps taken
    double reward = 0;
    bool done = false;
    int steps = 0;
    while (steps < xcsf->TELETRANSPORTATION && !done) {
        xcs_rl_init_step(xcsf);
        const double *state = env_get_state(xcsf);
        const int action = xcs_rl_decision(xcsf, state);
        reward = env_execute(xcsf, action);
        done = env_is_done(xcsf);
        xcs_rl_update(xcsf, state, action, reward, done);
        *error +=
            xcs_rl_error(xcsf, action, reward, done, env_max_payoff(xcsf));
        xcs_rl_end_step(xcsf, state, action, reward);
        ++steps;
    }
    xcs_rl_end_trial(xcsf);
    *error /= steps;
    if (!env_multistep(xcsf)) {
        return (reward > 0) ? 1 : 0;
    }
    return steps;
}

/**
 * @brief Executes a reinforcement learning experiment.
 * @param [in] xcsf The XCSF data structure.
 * @return The mean number of steps to goal.
 */
double
xcs_rl_exp(struct XCSF *xcsf)
{
    double error = 0; // prediction error: individual trial
    double werr = 0; // prediction error: windowed total
    double tperf = 0; // steps to goal: total over all trials
    double wperf = 0; // steps to goal: windowed total
    for (int cnt = 0; cnt < xcsf->MAX_TRIALS; ++cnt) {
        xcs_rl_trial(xcsf, &error, true); // explore
        const double perf = xcs_rl_trial(xcsf, &error, false); // exploit
        wperf += perf;
        tperf += perf;
        werr += error;
        perf_print(xcsf, &wperf, &werr, cnt);
    }
    return tperf / xcsf->MAX_TRIALS;
}

/**
 * @brief Creates and updates an action set for a given (state, action, reward).
 * @param [in] xcsf The XCSF data structure.
 * @param [in] state The input state to match.
 * @param [in] action The selected action.
 * @param [in] reward The reward for having performed the action.
 * @return The prediction error.
 */
double
xcs_rl_fit(struct XCSF *xcsf, const double *state, const int action,
           const double reward)
{
    xcs_rl_init_trial(xcsf);
    xcs_rl_init_step(xcsf);
    clset_match(xcsf, state);
    pa_build(xcsf, state);
    const double prediction = pa_val(xcsf, action);
    const double error = (xcsf->loss_ptr)(xcsf, &prediction, &reward);
    param_set_explore(xcsf, true); // ensure EA is executed
    xcs_rl_update(xcsf, state, action, reward, true);
    xcs_rl_end_step(xcsf, state, action, reward);
    xcs_rl_end_trial(xcsf);
    xcsf->error += (error - xcsf->error) * xcsf->BETA;
    return error;
}

/**
 * @brief Initialises a reinforcement learning trial.
 * @param [in] xcsf The XCSF data structure.
 */
void
xcs_rl_init_trial(struct XCSF *xcsf)
{
    xcsf->prev_reward = 0;
    xcsf->prev_pred = 0;
    if (xcsf->x_dim < 1) { // memory allocation guard
        printf("xcs_rl_init_trial(): error x_dim less than 1\n");
        xcsf->x_dim = 1;
        exit(EXIT_FAILURE);
    }
    xcsf->prev_state = malloc(sizeof(double) * xcsf->x_dim);
    clset_init(&xcsf->prev_aset);
    clset_init(&xcsf->kset);
}

/**
 * @brief Frees memory used by a reinforcement learning trial.
 * @param [in] xcsf The XCSF data structure.
 */
void
xcs_rl_end_trial(struct XCSF *xcsf)
{
    clset_free(&xcsf->prev_aset);
    clset_kill(xcsf, &xcsf->kset);
    free(xcsf->prev_state);
}

/**
 * @brief Initialises a step in a reinforcement learning trial.
 * @param [in] xcsf The XCSF data structure.
 */
void
xcs_rl_init_step(struct XCSF *xcsf)
{
    clset_init(&xcsf->mset);
    clset_init(&xcsf->aset);
}

/**
 * @brief Ends a step in a reinforcement learning trial.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] state The current input state.
 * @param [in] action The current action.
 * @param [in] reward The current reward.
 */
void
xcs_rl_end_step(struct XCSF *xcsf, const double *state, const int action,
                const double reward)
{
    clset_free(&xcsf->mset);
    clset_free(&xcsf->prev_aset);
    xcsf->prev_aset = xcsf->aset;
    xcsf->prev_reward = reward;
    xcsf->prev_pred = pa_val(xcsf, action);
    memcpy(xcsf->prev_state, state, sizeof(double) * xcsf->x_dim);
}

/**
 * @brief Provides reinforcement to the sets.
 * @details Creates the action set, updates the classifiers and runs the EA.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] state The input state.
 * @param [in] action The action selected.
 * @param [in] reward The reward from performing the action.
 * @param [in] done Whether the environment is in a terminal state.
 */
void
xcs_rl_update(struct XCSF *xcsf, const double *state, const int action,
              const double reward, const bool done)
{
    clset_action(xcsf, action); // create action set
    if (xcsf->prev_aset.list != NULL) { // update previous action set and run EA
        const double p = xcsf->prev_reward + (xcsf->GAMMA * pa_best_val(xcsf));
        clset_validate(&xcsf->prev_aset);
        clset_update(xcsf, &xcsf->prev_aset, xcsf->prev_state, &p, false);
        if (xcsf->explore) {
            ea(xcsf, &xcsf->prev_aset);
        }
    }
    if (done) { // in terminal state: update current action set and run EA
        clset_validate(&xcsf->aset);
        clset_update(xcsf, &xcsf->aset, state, &reward, true);
        if (xcsf->explore) {
            ea(xcsf, &xcsf->aset);
        }
    }
}

/**
 * @brief Returns the reinforcement learning system prediction error.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] action The current action.
 * @param [in] reward The current reward.
 * @param [in] done Whether the environment is in a terminal state.
 * @param [in] max_p The maximum payoff in the environment.
 * @return The prediction error.
 */
double
xcs_rl_error(struct XCSF *xcsf, const int action, const double reward,
             const bool done, const double max_p)
{
    double error = 0;
    const double prediction = pa_val(xcsf, action);
    if (xcsf->prev_aset.list != NULL) {
        const double p = xcsf->prev_reward + (xcsf->GAMMA * prediction);
        error += (xcsf->loss_ptr)(xcsf, &xcsf->prev_pred, &p) / max_p;
    }
    if (done) {
        error += (xcsf->loss_ptr)(xcsf, &prediction, &reward) / max_p;
    }
    xcsf->error += (error - xcsf->error) * xcsf->BETA;
    return error;
}

/**
 * @brief Selects an action to perform in a reinforcement learning problem.
 * @details Constructs the match set and selects an action to perform.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] state The input state.
 * @return The selected action.
 */
int
xcs_rl_decision(struct XCSF *xcsf, const double *state)
{
    clset_match(xcsf, state);
    pa_build(xcsf, state);
    if (xcsf->explore && rand_uniform(0, 1) < xcsf->P_EXPLORE) {
        return pa_rand_action(xcsf);
    }
    return pa_best_action(xcsf);
}
