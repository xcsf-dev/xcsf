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

#ifdef GPU
#include "cuda.h"
#endif

static double xcs_single_trial(XCSF *xcsf, double *error, _Bool explore);

/**
 * @brief Executes a single-step reinforcement learning experiment.
 * @param xcsf The XCSF data structure.
 * @return The mean accuracy.
 */
double xcs_single_step_exp(XCSF *xcsf)
{
#ifdef GPU
    CUDA_CALL( cudaMalloc((void **) &xcsf->x_gpu, xcsf->x_dim * sizeof(double)) );
    CUDA_CALL( cudaMalloc((void **) &xcsf->y_gpu, xcsf->y_dim * sizeof(double)) );
#endif
    pa_init(xcsf);
    double error = 0; // prediction error: individual trial
    double werr = 0; // prediction error: windowed total
    double tperf = 0; // accuracy: total over all trials
    double wperf = 0; // accuracy: windowed total
    for(int cnt = 0; cnt < xcsf->MAX_TRIALS; cnt++) {
        xcs_single_trial(xcsf, &error, true); // explore
        double perf = xcs_single_trial(xcsf, &error, false); // exploit
        wperf += perf;
        tperf += perf;
        werr += error;
        perf_print(xcsf, &wperf, &werr, cnt);
    }
    pa_free(xcsf);
#ifdef GPU
    CUDA_CALL( cudaFree(xcsf->x_gpu) );
    CUDA_CALL( cudaFree(xcsf->y_gpu) );
#endif
    return tperf / xcsf->MAX_TRIALS;
}                                

/**
 * @brief Executes a single-step trial using a built-in environment.
 * @param xcsf The XCSF data structure.
 * @param error The system prediction error (set by this function).
 * @param explore Whether this is an exploration or exploitation trial.
 * @return The classification accuracy.
 */
static double xcs_single_trial(XCSF *xcsf, double *error, _Bool explore)
{
    xcs_single_init(xcsf);
    xcsf->train = explore;
    const double *x = env_get_state(xcsf);
    int action = xcs_single_decision(xcsf, x);
    double reward = env_execute(xcsf, action);
    *error = xcs_single_error(xcsf, reward);
    if(explore) {
        xcs_single_update(xcsf, x, action, reward);
    }
    xcs_single_free(xcsf);
    return (reward > 0) ? 1 : 0;
}

/**
 * @brief Initialises match, action, and kill sets.
 * @param xcsf The XCSF data structure.
 */
void xcs_single_init(XCSF *xcsf) {
    clset_init(&xcsf->mset);
    clset_init(&xcsf->aset);
    clset_init(&xcsf->kset);
}

/**
 * @brief Frees memory used by match, action, and kill sets.
 * @param xcsf The XCSF data structure.
 */
void xcs_single_free(XCSF *xcsf)
{
    clset_free(&xcsf->mset);
    clset_free(&xcsf->aset);
    clset_kill(xcsf, &xcsf->kset);
}

/**
 * @brief Constructs the match set and selects an action to perform.
 * @param xcsf The XCSF data structure.
 * @param x The input state.
 * @return The selected action.
 */
int xcs_single_decision(XCSF *xcsf, const double *x)
{
#ifdef GPU
    CUDA_CALL(cudaMemcpy(xcsf->x_gpu, x, xcsf->x_dim * sizeof(double), cudaMemcpyHostToDevice));
#endif
    clset_match(xcsf, x);
    pa_build(xcsf, x);
    if(xcsf->train && rand_uniform(0,1) < xcsf->P_EXPLORE) {
        return pa_rand_action(xcsf);
    }
    return pa_best_action(xcsf);
}

/**
 * @brief Creates the action set, updates the classifiers and runs the EA.
 * @param xcsf The XCSF data structure.
 * @param x The input state.
 * @param action The action selected.
 * @param reward The reward from performing the action.
 */
void xcs_single_update(XCSF *xcsf, const double *x, int action, double reward)
{
#ifdef GPU
    CUDA_CALL(cudaMemcpy(xcsf->y_gpu, &reward, sizeof(double), cudaMemcpyHostToDevice));
#endif
    clset_action(xcsf, action);
    clset_update(xcsf, &xcsf->aset, x, &reward, true);
    ea(xcsf, &xcsf->aset);
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
