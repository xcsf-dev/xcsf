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
 * @file xcs_single_step.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Single-step reinforcement learning functions.
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
#include "xcs_single_step.h"
#include "env.h"

static double xcs_single_trial(XCSF *xcsf, double *perf, _Bool explore);

/**
 * @brief Executes a single-step experiment.
 * @param xcsf The XCSF data structure.
 * @return The mean prediction error.
 */
double xcs_single_step_exp(XCSF *xcsf)
{
    gplot_init(xcsf);
    pa_init(xcsf);
    double err = 0; // total error over all trials
    double perr = 0; // windowed accuracy for averaging
    double pterr = 0; // windowed prediction error for averaging
    double perf = 0; // individual trial accuracy
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        xcs_single_trial(xcsf, &perf, true); // explore
        double error = xcs_single_trial(xcsf, &perf, false); // exploit
        perr += perf;
        err += error;
        pterr += error;
        if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
            disp_perf2(xcsf, perr/xcsf->PERF_AVG_TRIALS, pterr/xcsf->PERF_AVG_TRIALS, cnt);
            perr = 0;
            pterr = 0;
        }
    }
    gplot_free(xcsf);
    pa_free(xcsf);
    return err / xcsf->MAX_TRIALS;
}                                

/**
 * @brief Executes a single-step trial using a built-in environment.
 * @param xcsf The XCSF data structure.
 * @param perf The classification accuracy (set by this function).
 * @param explore Whether this is an exploration or exploitation trial.
 * @return The system prediction error.
 */
static double xcs_single_trial(XCSF *xcsf, double *perf, _Bool explore)
{
    SET mset; // match set
    SET aset; // action set
    SET kset; // kill set
    clset_init(xcsf, &mset);
    clset_init(xcsf, &aset);
    clset_init(xcsf, &kset);
    xcsf->train = explore;
    const double *x = env_get_state(xcsf);
    int action = xcs_single_decision(xcsf, &mset, &kset, x);
    double reward = env_execute(xcsf, action);
    if(reward > 0) {
        *perf = 1;
    }
    else {
        *perf = 0;
    }
    if(explore) {
        xcs_single_update(xcsf, &mset, &aset, &kset, x, action, reward);
    }
    clset_kill(xcsf, &kset);
    clset_free(xcsf, &aset);
    clset_free(xcsf, &mset);
    return xcs_single_error(xcsf, reward);
}

/**
 * @brief Constructs the match set and selects an action to perform.
 * @param xcsf The XCSF data structure.
 * @param mset The match set.
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 * @param x The input state.
 * @return The selected action.
 */
int xcs_single_decision(XCSF *xcsf, SET *mset, SET *kset, const double *x)
{
    clset_match(xcsf, mset, kset, x);
    xcsf->msetsize += (mset->size - xcsf->msetsize) * xcsf->BETA;
    pa_build(xcsf, mset, x);
    if(xcsf->train && rand_uniform(0,1) < xcsf->P_EXPLORE) {
        return pa_rand_action(xcsf);
    }
    return pa_best_action(xcsf);
}

/**
 * @brief Creates the action set, updates the classifiers and runs the EA.
 * @param xcsf The XCSF data structure.
 * @param mset The match set.
 * @param aset The action set.
 * @param kset A set to store deleted macro-classifiers for later memory removal.
 * @param x The input state.
 * @param a The action selected.
 * @param r The reward from performing the action.
 */
void xcs_single_update(XCSF *xcsf, const SET *mset, SET *aset, SET *kset, const double *x, int a, double r)
{
    clset_action(xcsf, mset, aset, a);
    clset_update(xcsf, aset, kset, x, &r, true);
    ea(xcsf, aset, kset);
    xcsf->time += 1;
}

/**
 * @brief Returns the system error.
 * @param xcsf The XCSF data structure.
 * @param reward The reward from performing the action.
 * @return The system prediction error.
 */
double xcs_single_error(const XCSF *xcsf, double reward)
{
    return fabs(reward - pa_best_val(xcsf));
}
