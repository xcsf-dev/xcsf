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

#include "clset.h"
#include "config.h"
#include "env_csv.h"
#include "pa.h"
#include "param.h"
#include "utils.h"
#include "xcs_rl.h"
#include "xcs_supervised.h"
#include "xcsf.h"

int
main(int argc, char **argv)
{
    if (argc < 3 || argc > 5) {
        printf("Usage: xcsf problemType{csv|mp|maze} ");
        printf("problem{.csv|size|maze} [config.ini] [xcs.bin]\n");
        exit(EXIT_FAILURE);
    }
    struct XCSF *xcsf = malloc(sizeof(struct XCSF));
    rand_init();
    env_init(xcsf, argv); // initialise environment and default parameters
    if (argc > 3) { // load parameter config
        config_read(xcsf, argv[3]);
    } else {
        config_read(xcsf, "default.ini");
    }
    xcsf_init(xcsf); // initialise empty sets
    if (argc == 5) { // reload state of a previous experiment
        const size_t s = xcsf_load(xcsf, argv[4]);
        printf("XCSF loaded: %d elements\n", (int) s);
    } else { // new experiment
        clset_pset_init(xcsf);
    }
    pa_init(xcsf); // initialise prediction array
    param_print(xcsf); // print parameters used
    if (strcmp(argv[1], "csv") == 0) { // supervised regression - csv file
        const struct EnvCSV *env = xcsf->env;
        xcs_supervised_fit(xcsf, env->train_data, env->test_data, true);
    } else { // reinforcement learning - maze or mux
        xcs_rl_exp(xcsf);
    }
    pa_free(xcsf); // clean up
    env_free(xcsf);
    xcsf_free(xcsf);
    param_free(xcsf);
    free(xcsf);
    return EXIT_SUCCESS;
}
