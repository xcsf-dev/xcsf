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
 * @file main.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Main function for stand-alone binary execution.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "xcsf.h"
#include "utils.h"
#include "config.h"
#include "env.h"
#include "env_csv.h"
#include "clset.h"
#include "xcs_single_step.h"
#include "xcs_multi_step.h"

int main(int argc, char **argv)
{    
    if(argc < 3 || argc > 5) {
        printf("Usage: xcsf problemType{csv|mp|maze} problem{.csv|size|maze} [config.ini] [xcs.bin]\n");
        exit(EXIT_FAILURE);
    } 
    XCSF *xcsf = malloc(sizeof(XCSF));
    random_init();
    // load parameter config
    if(argc > 3) {
        config_init(xcsf, argv[3]);
    }
    else {
        config_init(xcsf, "default.ini");
    }
    // initialise problem environment
    env_init(xcsf, argv);
    // initialise empty sets
    xcsf_init(xcsf);
    // reload state of a previous experiment
    if(argc == 5) {
        printf("LOADING XCSF\n");
        xcsf_load(xcsf, argv[4]);
    }
    // new experiment
    else {
        clset_pop_init(xcsf);
    }
    // supervised regression - input csv file
    if(strcmp(argv[1], "csv") == 0) {
        const ENV_CSV *env = xcsf->env;
        xcsf_fit2(xcsf, env->train_data, env->test_data, true);
    }
    // reinforcement learning - maze or mux
    else {
        if(env_multistep(xcsf)) {
            xcs_multi_step_exp(xcsf);
        }
        else {
            xcs_single_step_exp(xcsf);
        }
    }
    // clean up
    env_free(xcsf);
    clset_kill(xcsf, &xcsf->pset);
    config_free(xcsf);
    free(xcsf);
    return EXIT_SUCCESS;
}
