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
 * @file xcs_multi_step.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Multi-step reinforcement learning functions.
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include "xcsf.h"
#include "utils.h"
#include "param.h"
#include "cl.h"
#include "clset.h"
#include "pa.h"
#include "ea.h"
#include "perf.h"
#include "xcs_multi_step.h"
#include "env.h"

static double xcs_multi_trial(XCSF *xcsf, double *error, _Bool explore);

/**
 * @brief Executes a multi-step reinforcement learning experiment.
 * @param xcsf The XCSF data structure.
 * @return The mean number of steps to goal.
 */
double xcs_multi_step_exp(XCSF *xcsf)
{
    pa_init(xcsf);
    double error = 0; // prediction error: individual trial
    double werr = 0; // prediction error: windowed total
    double tperf = 0; // steps to goal: total over all trials
    double wperf = 0; // steps to goal: windowed total
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        xcs_multi_trial(xcsf, &error, true); // explore
        double perf = xcs_multi_trial(xcsf, &error, false); // exploit
        wperf += perf;
        tperf += perf;
        werr += error;
        perf_print(xcsf, &wperf, &werr, cnt);
    }
    pa_free(xcsf);
    return tperf / xcsf->MAX_TRIALS;
}                                

/**
 * @brief Executes a multi-step trial using a built-in environment.
 * @param xcsf The XCSF data structure.
 * @param error The mean system prediction error (set by this function).
 * @param explore Whether this is an exploration or exploitation trial.
 * @return The number of steps taken to reach the goal.
 */
static double xcs_multi_trial(XCSF *xcsf, double *error, _Bool explore)
{
    if(xcsf->x_dim < 1) { // memory allocation guard
        printf("xcs_multi_trial(): error x_dim less than 1\n");
        xcsf->x_dim = 1;
        exit(EXIT_FAILURE);
    }
    double *prev_state = malloc(xcsf->x_dim * sizeof(double));
    env_reset(xcsf);
    param_set_train(xcsf, explore);
    _Bool reset = false; 
    double prev_reward = 0;
    double prev_pred = 0;
    clset_init(&xcsf->prev_aset);
    clset_init(&xcsf->kset);
    *error = 0;
    int steps = 0;
    for(steps = 0; steps < xcsf->TELETRANSPORTATION && !reset; steps++) {
        clset_init(&xcsf->mset);
        clset_init(&xcsf->aset);
        const double *state = env_get_state(xcsf);
        int action = xcs_multi_decision(xcsf, state);
        clset_action(xcsf, action);
        double reward = env_execute(xcsf, action);
        reset = env_is_reset(xcsf);
        // update previous action set and run EA
        if(xcsf->prev_aset.list != NULL) {
            double payoff = prev_reward + (xcsf->GAMMA * pa_best_val(xcsf));
            clset_validate(&xcsf->prev_aset);
            clset_update(xcsf, &xcsf->prev_aset, prev_state, &payoff, false);
            if(xcsf->train) {
                ea(xcsf, &xcsf->prev_aset);
            }
            *error += fabs(xcsf->GAMMA * pa_val(xcsf, action) 
                    + prev_reward - prev_pred) / env_max_payoff(xcsf);
        }
        // in goal state: update current action set and run EA
        if(reset) {
            clset_validate(&xcsf->aset);
            clset_update(xcsf, &xcsf->aset, state, &reward, true);
            if(xcsf->train) {
                ea(xcsf, &xcsf->aset);
            }
            *error += fabs(reward - pa_val(xcsf, action)) / env_max_payoff(xcsf);
        }
        // next step
        clset_free(&xcsf->mset);
        clset_free(&xcsf->prev_aset);
        xcsf->prev_aset = xcsf->aset;
        prev_reward = reward;
        prev_pred = pa_val(xcsf, action);
        memcpy(prev_state, state, sizeof(double) * xcsf->x_dim);
    }
    clset_free(&xcsf->prev_aset);
    clset_kill(xcsf, &xcsf->kset);
    free(prev_state);
    *error /= steps;
    return steps;
}

/**
 * @brief Constructs the match set and selects an action to perform.
 * @param xcsf The XCSF data structure.
 * @param x The input state.
 * @return The selected action.
 */
int xcs_multi_decision(XCSF *xcsf, const double *x)
{
    clset_match(xcsf, x);
    pa_build(xcsf, x);
    if(xcsf->train && rand_uniform(0,1) < xcsf->P_EXPLORE) {
        return pa_rand_action(xcsf);
    }
    return pa_best_action(xcsf);
}
