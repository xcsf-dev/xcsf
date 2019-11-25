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
 * @date 2015--2019.
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
#include "cl_set.h"
#include "pa.h"
#include "ea.h"
#include "perf.h"
#include "xcs_multi_step.h"
#include "env.h"

int xcs_explore_multi(XCSF *xcsf);
int xcs_exploit_multi(XCSF *xcsf, double *error);

/**
 * @brief Executes a multi-step reinforcement learning experiment.
 * @param xcsf The XCSF data structure.
 */
double xcs_multi_step_exp(XCSF *xcsf)
{
    gplot_init(xcsf);
    pa_init(xcsf);
    double perr = 0, err = 0, pterr = 0;
    xcsf->train = true;
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        xcs_explore_multi(xcsf);
        perr += xcs_exploit_multi(xcsf, &pterr);
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
 * @brief Executes a multi-step explore trial.
 * @param xcsf The XCSF data structure.
 * @return The number of steps taken to reach the goal.
 */
int xcs_explore_multi(XCSF *xcsf)
{
    _Bool reset = false; 
    double prev_reward = 0;
    double *prev_state = malloc(sizeof(double) * xcsf->num_x_vars);
    SET prev_aset, kset;
    set_init(xcsf, &prev_aset);
    set_init(xcsf, &kset);
    env_reset(xcsf);
    int steps = 0;
    for(steps = 0; steps < xcsf->TELETRANSPORTATION && !reset; steps++) {
        SET mset, aset;
        set_init(xcsf, &mset);
        set_init(xcsf, &aset);
        // percieve environment
        double *state = env_get_state(xcsf);
        // generate match set
        set_match(xcsf, &mset, &kset, state);
        pa_build(xcsf, &mset, state);
        // select a random move
        int action = pa_rand_action(xcsf);
        // generate action set
        set_action(xcsf, &mset, &aset, action);
        // get environment feedback
        double reward = env_execute(xcsf, action);
        reset = env_is_reset(xcsf);
        // update previous action set and run EA
        if(prev_aset.list != NULL) {
            double payoff = prev_reward + (xcsf->GAMMA * pa_best_val(xcsf));
            set_validate(xcsf, &prev_aset);
            set_update(xcsf, &prev_aset, &kset, prev_state, &payoff);
            ea(xcsf, &prev_aset, &kset);
        }
        // in goal state: update current action set and run EA
        if(reset) {
            set_validate(xcsf, &aset);
            set_update(xcsf, &aset, &kset, state, &reward);
            ea(xcsf, &aset, &kset);
        }
        // next step
        xcsf->time += 1;
        xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
        set_free(xcsf, &mset); // frees the match set list
        set_free(xcsf, &prev_aset); // frees the previous action set list
        prev_aset = aset;
        prev_reward = reward;
        memcpy(prev_state, state, sizeof(double) * xcsf->num_x_vars);
    }
    set_free(xcsf, &prev_aset); // frees the previous action set list
    set_kill(xcsf, &kset); // kills deleted classifiers
    free(prev_state);
    return steps;
}

/**
 * @brief Executes a multi-step exploit trial.
 * @param xcsf The XCSF data structure.
 * @param error The prediction error (set by this function).
 * @return The number of steps taken to reach the goal.
 */
int xcs_exploit_multi(XCSF *xcsf, double *error)
{
    _Bool reset = false; 
    double prev_reward = 0, prev_pred = 0;
    double *prev_state = malloc(sizeof(double) * xcsf->num_x_vars);
    SET prev_aset, kset;
    set_init(xcsf, &prev_aset);
    set_init(xcsf, &kset);
    *error = 0;
    env_reset(xcsf);
    int steps = 0;
    for(steps = 0; steps < xcsf->TELETRANSPORTATION && !reset; steps++) {
        SET mset, aset;
        set_init(xcsf, &mset);
        set_init(xcsf, &aset);
        // percieve environment
        double *state = env_get_state(xcsf);
        // generate match set
        set_match(xcsf, &mset, &kset, state);
        // select the best move
        pa_build(xcsf, &mset, state);
        int action = pa_best_action(xcsf);
        // generate action set
        set_action(xcsf, &mset, &aset, action);
        // get environment feedback
        double reward = env_execute(xcsf, action);
        reset = env_is_reset(xcsf);
        // update previous action set
        if(prev_aset.list != NULL) {
            set_validate(xcsf, &prev_aset);
            double payoff = prev_reward + (xcsf->GAMMA * pa_best_val(xcsf));
            set_update(xcsf, &prev_aset, &kset, prev_state, &payoff);
            //ea(xcsf, &prev_aset, &kset);
            *error += fabs(xcsf->GAMMA * pa_val(xcsf, action) 
                    + prev_reward - prev_pred) / env_max_payoff(xcsf);
        }
        // in goal state: update current action set
        if(reset) {
            set_validate(xcsf, &aset);
            set_update(xcsf, &aset, &kset, state, &reward);
            //ea(xcsf, &aset, &kset);
            *error += fabs(reward - pa_val(xcsf, action)) / env_max_payoff(xcsf);
        }
        // next step
        //xcsf->time += 1;
        xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
        set_free(xcsf, &prev_aset); // frees the previous action set list
        set_free(xcsf, &mset); // frees the match set list
        prev_aset = aset;
        prev_reward = reward;
        prev_pred = pa_val(xcsf, action);
        memcpy(prev_state, state, sizeof(double) * xcsf->num_x_vars);
    }
    set_free(xcsf, &prev_aset); // frees the previous action set list
    set_kill(xcsf, &kset); // kills deleted classifiers
    free(prev_state);
    return steps;
}
