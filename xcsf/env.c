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
 * @file env.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief Built-in problem environment interface.
 */

#include "env_csv.h"
#include "env_maze.h"
#include "env_mux.h"

/**
 * @brief Initialises a built-in problem environment.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] argv The command line arguments.
 */
void
env_init(struct XCSF *xcsf, char **argv)
{
    char *end = NULL;
    if (strcmp(argv[1], "mp") == 0) {
        xcsf->env_vptr = &env_mux_vtbl;
        env_mux_init(xcsf, (int) strtoimax(argv[2], &end, 10));
    } else if (strcmp(argv[1], "maze") == 0) {
        xcsf->env_vptr = &env_maze_vtbl;
        env_maze_init(xcsf, argv[2]);
    } else if (strcmp(argv[1], "csv") == 0) {
        xcsf->env_vptr = &env_csv_vtbl;
        env_csv_init(xcsf, argv[2]);
    } else {
        printf("Invalid environment specified: %s\n", argv[1]);
        printf("Available environments: {mp, maze, csv}\n");
        exit(EXIT_FAILURE);
    }
}
