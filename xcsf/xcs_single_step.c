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
 * @file xcs_single_step.c
 * @brief Single-step classification functions.
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

void xcs_explore_single(XCSF *xcsf);
double xcs_exploit_single(XCSF *xcsf, double *error);

double xcs_single_step_exp(XCSF *xcsf)
{
    gplot_init(xcsf);
    pa_init(xcsf);
    double perr = 0, err = 0, pterr = 0;
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        xcs_explore_single(xcsf);
        double error = 0;
        perr += xcs_exploit_single(xcsf, &error);
        err += error;
        pterr += error;
        if(cnt % xcsf->PERF_AVG_TRIALS == 0 && cnt > 0) {
            disp_perf2(xcsf, perr/xcsf->PERF_AVG_TRIALS, pterr/xcsf->PERF_AVG_TRIALS, cnt);
            perr = 0; pterr = 0;
        }
    }
    gplot_free(xcsf);
    pa_free(xcsf);
    return err/xcsf->MAX_TRIALS;  
}                                

void xcs_explore_single(XCSF *xcsf)
{
    xcsf->train = true;
    double *x = env_get_state(xcsf);
    SET mset, aset, kset;
    set_init(xcsf, &mset);
    set_init(xcsf, &aset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, x);
    pa_build(xcsf, &mset, x);
    int action = pa_rand_action(xcsf);
    double reward = env_execute(xcsf, action);
    set_action(xcsf, &mset, &aset, action);
    set_update(xcsf, &aset, &kset, x, &reward);
    ea(xcsf, &aset, &kset);
    xcsf->time += 1;
    xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
    set_kill(xcsf, &kset); // kills deleted classifiers
    set_free(xcsf, &aset); // frees the action set list
    set_free(xcsf, &mset); // frees the match set list
}
 
double xcs_exploit_single(XCSF *xcsf, double *error)
{
    xcsf->train = false;
    double *x = env_get_state(xcsf);
    SET mset, aset, kset;
    set_init(xcsf, &mset);
    set_init(xcsf, &aset);
    set_init(xcsf, &kset);
    set_match(xcsf, &mset, &kset, x);
    pa_build(xcsf, &mset, x);
    int action = pa_best_action(xcsf);
    double reward = env_execute(xcsf, action);
    set_action(xcsf, &mset, &aset, action);
    xcsf->msetsize += (mset.size - xcsf->msetsize) * xcsf->BETA;
    set_kill(xcsf, &kset); // kills deleted classifiers
    set_free(xcsf, &aset); // frees the action set list
    set_free(xcsf, &mset); // frees the match set list
    *error = fabs(reward - pa_best_val(xcsf));
    // return classification accuracy
    if(reward > 0) {
        return 1;
    }
    return 0;
}
