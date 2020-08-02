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
 * @file env_maze.h
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2020.
 * @brief The discrete maze problem environment module.
 */

#pragma once

#include "env.h"
#include "xcsf.h"

#define MAX_SIZE (50) //!< The maximum width/height of a maze

/**
 * @brief Maze environment data structure.
 */
typedef struct ENV_MAZE {
    double *state; //!< Current state
    char maze[MAX_SIZE][MAX_SIZE]; //!< Maze
    int xpos; //!< Current x position
    int ypos; //!< Current y position
    int xsize; //!< Maze size in x dimension
    int ysize; //!< Maze size in y dimension
    _Bool reset; //!< Whether the trial needs to be reset (e.g., in goal state)
} ENV_MAZE;

_Bool
env_maze_isreset(const struct XCSF *xcsf);

_Bool
env_maze_multistep(const struct XCSF *xcsf);

double
env_maze_execute(const struct XCSF *xcsf, int action);

double
env_maze_maxpayoff(const struct XCSF *xcsf);

const double *
env_maze_get_state(const struct XCSF *xcsf);

void
env_maze_free(const struct XCSF *xcsf);

void
env_maze_init(struct XCSF *xcsf, const char *filename);

void
env_maze_reset(const struct XCSF *xcsf);

/**
 * @brief Maze environment implemented functions.
 */
static struct EnvVtbl const env_maze_vtbl = {
    &env_maze_isreset,   &env_maze_multistep, &env_maze_execute,
    &env_maze_maxpayoff, &env_maze_get_state, &env_maze_free,
    &env_maze_reset};
