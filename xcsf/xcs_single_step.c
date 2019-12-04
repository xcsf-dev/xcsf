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
 * @date 2015--2019.
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
#include "cl_set.h"
#include "pa.h"
#include "ea.h"
#include "perf.h"
#include "xcs_single_step.h"
#include "env.h"

void xcs_single_explore(XCSF *xcsf);
double xcs_single_exploit(XCSF *xcsf, double *error);

/**
 * @brief Executes a single-step experiment.
 * @param xcsf The XCSF data structure.
 * @return The mean prediction error.
 */
double xcs_single_step_exp(XCSF *xcsf)
{
    gplot_init(xcsf);
    pa_init(xcsf);
    double perr = 0;
    double err = 0;
    double pterr = 0;
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        xcs_single_explore(xcsf);
        double error = 0;
        perr += xcs_single_exploit(xcsf, &error);
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
 * @brief Executes a single-step explore trial.
 * @param xcsf The XCSF data structure.
 */
void xcs_single_explore(XCSF *xcsf)
{
    xcsf->train = true;
    double *x = env_get_state(xcsf);
    SET mset; // match set
    SET aset; // action set
    SET kset; // kill set
    set_init(xcsf, &mset);
    set_init(xcsf, &aset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, x);
    pa_build(xcsf, &mset, x);
    int action = 0;
    if(rand_uniform(0,1) < xcsf->P_EXPLORE) {
        action = pa_rand_action(xcsf);
    }
    else {
        action = pa_best_action(xcsf);
    }
    double reward = env_execute(xcsf, action);
    set_action(xcsf, &mset, &aset, action);
    set_update(xcsf, &aset, &kset, x, &reward, true);
    ea(xcsf, &aset, &kset);
    xcsf->time += 1;
    xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
    set_kill(xcsf, &kset);
    set_free(xcsf, &aset);
    set_free(xcsf, &mset);
}

/**
 * @brief Executes a single-step exploit trial.
 * @param xcsf The XCSF data structure.
 * @param error The prediction error (set by this function).
 * @return Whether the correct action was selected.
 */
double xcs_single_exploit(XCSF *xcsf, double *error)
{
    xcsf->train = false;
    double *x = env_get_state(xcsf);
    SET mset; // match set
    SET aset; // action set
    SET kset; // kill set
    set_init(xcsf, &mset);
    set_init(xcsf, &aset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, x);
    pa_build(xcsf, &mset, x);
    int action = pa_best_action(xcsf);
    double reward = env_execute(xcsf, action);
    set_action(xcsf, &mset, &aset, action);
    xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
    set_kill(xcsf, &kset);
    set_free(xcsf, &aset);
    set_free(xcsf, &mset);
    *error = fabs(reward - pa_best_val(xcsf));
    if(reward > 0) {
        return 1;
    }
    return 0;
}
