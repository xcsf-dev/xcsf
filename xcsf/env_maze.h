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
 * @date 2015--2019.
 * @brief The discrete maze problem environment module.
 *
 * @details Reads in the chosen maze from a file where each entry specifies a
 * distinct position in the maze. The maze is toroidal and if the animat
 * reaches one edge it can reenter the maze from the other side. Obstacles are
 * coded as 'O' and 'Q', empty positions as '*', and food as 'F' or 'G'. The 8
 * adjacent cells are percevied (encoded as reals) and 8 movements are possible
 * to the adjacent cells (if not blocked.) The animat is initially placed at a
 * random empty position. The goal is to find the shortest path to the food.
 *
 * Some mazes require a form of memory to be solved optimally.
 * The optimal average number of steps for each maze is:
 *
 * Woods 1: 1.7
 * Woods 2: 1.7
 * Woods 14: 9.5
 * Maze 4: 3.5
 * Maze 5: 4.61
 * Maze 6: 5.19
 * Maze 7: 4.33
 * Maze 10: 5.11
 * Woods 101: 2.9
 * Woods 101 1/2: 3.1
 * Woods 102: 3.23
 */
    
#pragma once

_Bool env_maze_isreset(XCSF *xcsf);
_Bool env_maze_multistep(XCSF *xcsf);
double env_maze_execute(XCSF *xcsf, int action);
double env_maze_maxpayoff(XCSF *xcsf);
double *env_maze_get_state(XCSF *xcsf);
void env_maze_free(XCSF *xcsf);
void env_maze_init(XCSF *xcsf, char *filename);
void env_maze_reset(XCSF *xcsf);

static struct EnvVtbl const env_maze_vtbl = {
    &env_maze_isreset,
    &env_maze_multistep,
    &env_maze_execute,
    &env_maze_maxpayoff,
    &env_maze_get_state,
    &env_maze_free,
    &env_maze_reset
};      
