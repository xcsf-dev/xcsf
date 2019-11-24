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
 * @file env_maze.c
 * @brief The discrete maze problem environment module.
 *
 * @details Reads in the chosen maze from a file where each entry specifies a
 * distinct position in the maze. The maze is toroidal and if the animat
 * reaches one edge it can reenter the maze from the other side. Obstacles are
 * coded as 'O' and 'Q', empty positions as '*', and food as 'F' or 'G'. A 2
 * bit or 3 bit encoding is automatically chosen depending on the number of
 * perceptions. 8 movements are possible to adjacent cells (if not blocked.)
 * The animat is initially placed at a random empty position. The goal is to
 * find the shortest path to the food. 
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <errno.h>
#include "xcsf.h"
#include "utils.h"
#include "cl.h"
#include "cl_set.h"
#include "env.h"
#include "env_maze.h"
     
/**
 * @brief Maze environment data structure.
 */ 
typedef struct ENV_MAZE {
    double *state; //!< current state
    char maze[50][50]; //!< maze
    int xpos; //!< current x position
    int ypos; //!< current y position
    int xsize; //!< maze size in x dimension
    int ysize; //!< maze size in y dimension
    _Bool reset;
} ENV_MAZE;
 
#define MAX_PAYOFF 1000.0
const int x_moves[] ={ 0, +1, +1, +1,  0, -1, -1, -1}; 
const int y_moves[] ={-1, -1,  0, +1, +1, +1,  0, -1};

void env_maze_sensor(XCSF *xcsf, char s, double *dec);

void env_maze_init(XCSF *xcsf, char *filename)
{
    // open maze file
    FILE *fp = fopen(filename, "rt");
    if(fp == 0) {
        printf("could not open %s. %s.\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }
    // read maze
    ENV_MAZE *env = malloc(sizeof(ENV_MAZE));
    int c; int x = 0; int y = 0;
    while((c = fgetc(fp)) != EOF) {
        switch(c) {
            case '\n':
                y++;
                env->xsize = x;
                x = 0;
                break;
            default:
                env->maze[y][x] = c;
                x++;
                break;
        }
    }
    env->ysize = y;
    env->state = malloc(sizeof(double) * 8);
    xcsf->num_classes = 8;
    xcsf->num_x_vars = 8;
    xcsf->num_y_vars = 1;
    xcsf->env = env;
    fclose(fp);
    printf("Loaded MAZE = %s\n", filename);
}

void env_maze_free(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    free(env->state);
    free(env);
}

void env_maze_reset(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    env->reset = false;
    do {
        env->xpos = irand_uniform(0, env->xsize);
        env->ypos = irand_uniform(0, env->ysize);
    } while(env->maze[env->ypos][env->xpos] != '*');
}

_Bool env_maze_isreset(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    return env->reset;
}

double *env_maze_get_state(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    int spos = 0;
    for(int x = -1; x < 2; x++) {
        for(int y = -1; y < 2; y++) {
            // ignore current pos
            if(x == 0 && y == 0) {
                continue;
            }
            // toroidal maze
            char s = env->maze
                [(env->ysize - (env->ypos + y)) % env->ysize]
                [(env->xsize - (env->xpos + x)) % env->xsize];
            // convert sensor to real number
            env_maze_sensor(xcsf, s, &env->state[spos++]);
        }
    }
    return env->state;
}

void env_maze_sensor(XCSF *xcsf, char s, double *dec)
{
    (void)xcsf;
    switch(s) {
        case '*': *dec = 0.1; break;
        case 'O': *dec = 0.3; break;
        case 'G': *dec = 0.5; break;
        case 'F': *dec = 0.7; break;
        case 'Q': *dec = 0.9; break;
        default :
            printf("unsupported maze state\n");
            exit(EXIT_FAILURE);
    }
}

double env_maze_execute(XCSF *xcsf, int action)
{
    if(action < 0 || action > 7) {
        printf("invalid maze action\n");
        exit(EXIT_FAILURE);
    }
    ENV_MAZE *env = xcsf->env;
    // toroidal maze
    int newx = (env->xsize - (env->xpos + x_moves[action])) % env->xsize;
    int newy = (env->ysize - (env->ypos + y_moves[action])) % env->ysize;
    // make the move and recieve reward
    switch(env->maze[newy][newx]) {
        case '*':
            env->ypos = newy;
            env->xpos = newx;
            env->reset = false;
            return 0;
        case 'F': 
        case 'G':
            env->ypos = newy;
            env->xpos = newx;
            env->reset = true;
            return MAX_PAYOFF;
        case 'O': 
        case 'Q':
            env->reset = false;
            return 0;
        default:
            printf("invalid maze type\n");
            exit(EXIT_FAILURE);
    }
}
 
double env_maze_maxpayoff(XCSF *xcsf)
{
    (void)xcsf;
    return MAX_PAYOFF;
}
 
_Bool env_maze_multistep(XCSF *xcsf)
{
    (void)xcsf;
    return true;
}
