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
