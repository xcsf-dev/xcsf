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
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "clset.h"
#include "pa.h"
#include "ea.h"
#include "perf.h"
#include "xcs_multi_step.h"
#include "env.h"

static int xcs_multi_trial(XCSF *xcsf, double *error, _Bool explore);

/**
 * @brief Executes a multi-step reinforcement learning experiment.
 * @param xcsf The XCSF data structure.
 * @return The mean prediction error.
 */
double xcs_multi_step_exp(XCSF *xcsf)
{
    double perr = 0;
    double err = 0;
    double pterr = 0;
    gplot_init(xcsf);
    pa_init(xcsf);
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        xcs_multi_trial(xcsf, &pterr, true); // explore
        perr += xcs_multi_trial(xcsf, &pterr, false); // exploit
        err += pterr;
        if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
            disp_perf2(xcsf, perr/xcsf->PERF_AVG_TRIALS, pterr/xcsf->PERF_AVG_TRIALS, cnt);
            perr = 0; pterr = 0;
        }
    }
    gplot_free(xcsf);
    pa_free(xcsf);
    return err / xcsf->MAX_TRIALS;
}                                

/**
 * @brief Executes a multi-step trial using a built-in environment.
 * @param xcsf The XCSF data structure.
 * @param error The system prediction error (set by this function).
 * @param explore Whether this is an exploration or exploitation trial.
 * @return The number of steps taken to reach the goal.
 */
static int xcs_multi_trial(XCSF *xcsf, double *error, _Bool explore)
{
    env_reset(xcsf);
    xcsf->train = explore;
    _Bool reset = false; 
    double prev_reward = 0;
    double prev_pred = 0;
    double *prev_state = malloc(sizeof(double) * xcsf->num_x_vars);
    SET prev_aset; // previous action set
    SET kset; // kill set
    clset_init(xcsf, &prev_aset);
    clset_init(xcsf, &kset);
    *error = 0;
    int steps = 0;
    for(steps = 0; steps < xcsf->TELETRANSPORTATION && !reset; steps++) {
        SET mset; // match set
        SET aset; // action set
        clset_init(xcsf, &mset);
        clset_init(xcsf, &aset);
        // percieve environment
        const double *state = env_get_state(xcsf);
        // generate match set
        clset_match(xcsf, &mset, &kset, state);
        xcsf->msetsize += (mset.size - xcsf->msetsize) * (1 / (double) xcsf->PERF_AVG_TRIALS);
        xcsf->mfrac += (clset_mfrac(xcsf) - xcsf->mfrac) * (1 / (double) xcsf->PERF_AVG_TRIALS);
        // calculate the prediction array
        pa_build(xcsf, &mset, state);
        // select an action to perform
        int action = 0;
        if(xcsf->train && rand_uniform(0,1) < xcsf->P_EXPLORE) {
            action = pa_rand_action(xcsf);
        }
        else {
            action = pa_best_action(xcsf);
        }
        // generate action set
        clset_action(xcsf, &mset, &aset, action);
        // get environment feedback
        double reward = env_execute(xcsf, action);
        reset = env_is_reset(xcsf);
        // update previous action set and run EA
        if(prev_aset.list != NULL) {
            double payoff = prev_reward + (xcsf->GAMMA * pa_best_val(xcsf));
            clset_validate(xcsf, &prev_aset);
            clset_update(xcsf, &prev_aset, &kset, prev_state, &payoff, false);
            if(xcsf->train) {
                ea(xcsf, &prev_aset, &kset);
            }
            *error += fabs(xcsf->GAMMA * pa_val(xcsf, action) 
                    + prev_reward - prev_pred) / env_max_payoff(xcsf);
        }
        // in goal state: update current action set and run EA
        if(reset) {
            clset_validate(xcsf, &aset);
            clset_update(xcsf, &aset, &kset, state, &reward, true);
            if(xcsf->train) {
                ea(xcsf, &aset, &kset);
            }
            *error += fabs(reward - pa_val(xcsf, action)) / env_max_payoff(xcsf);
        }
        // next step
        if(xcsf->train) {
            xcsf->time += 1;
        }
        clset_free(xcsf, &mset); // frees the match set list
        clset_free(xcsf, &prev_aset); // frees the previous action set list
        prev_aset = aset;
        prev_reward = reward;
        prev_pred = pa_val(xcsf, action);
        memcpy(prev_state, state, sizeof(double) * xcsf->num_x_vars);
    }
    clset_free(xcsf, &prev_aset); // frees the previous action set list
    clset_kill(xcsf, &kset); // kills deleted classifiers
    free(prev_state);
    return steps;
}
