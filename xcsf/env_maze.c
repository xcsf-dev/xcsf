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
 * @file env_maze.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2019.
 * @brief The discrete maze problem environment module.
 *
 * @details Reads in the chosen maze from a file where each entry specifies a
 * distinct position in the maze. The maze is toroidal and if the animat
 * reaches one edge it can reenter the maze from the other side. Obstacles are
 * coded as 'O' and 'Q', empty positions as '*', and food as 'F' or 'G'. The 8
 * adjacent cells are perceived (encoded as reals) and 8 movements are possible
 * to the adjacent cells (if not blocked.) The animat is initially placed at a
 * random empty position. The goal is to find the shortest path to the food. 
 *
 * Some mazes require a form of memory to be solved optimally.
 * The optimal average number of steps for each maze is:
 *
 * Woods 1: 1.7 \n
 * Woods 2: 1.7 \n
 * Woods 14: 9.5 \n
 * Maze 4: 3.5 \n
 * Maze 5: 4.61 \n
 * Maze 6: 5.19 \n
 * Maze 7: 4.33 \n
 * Maze 10: 5.11 \n
 * Woods 101: 2.9 \n
 * Woods 101 1/2: 3.1 \n
 * Woods 102: 3.31
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

#define MAZE_DEBUG false //!< Whether to print the state of the maze during exploitation
#define MAX_SIZE 50 //!< The maximum width/height of a maze
#define MAX_PAYOFF 1.0 //!< The payoff provided at a food position
const int x_moves[] ={ 0, +1, +1, +1,  0, -1, -1, -1}; //!< Possible maze moves on x-axis
const int y_moves[] ={-1, -1,  0, +1, +1, +1,  0, -1}; //!< Possible maze moves on y-axis

void env_maze_print(XCSF *xcsf);

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

double env_maze_sensor(XCSF *xcsf, char s);

/**
 * @brief Initialises a maze environment from a specified file.
 * @param xcsf The XCSF data structure.
 * @param fname The file name of the specified maze environment.
 */
void env_maze_init(XCSF *xcsf, char *fname)
{
    // open maze file
    FILE *fp = fopen(fname, "rt");
    if(fp == 0) {
        printf("could not open %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    // read maze
    ENV_MAZE *env = malloc(sizeof(ENV_MAZE));
    int x = 0;
    int y = 0;
    int c;
    while((c = fgetc(fp)) != EOF) {
        if(c == '\n') {
            y++;
            env->xsize = x;
            x = 0;
        }
        else {
            env->maze[y][x] = (char)c;
            x++;
        }
        // check maximum maze size not exceeded
        if(x > MAX_SIZE || y > MAX_SIZE) {
            printf("Maze too big to be read. Max size = [%d,%d]\n", MAX_SIZE, MAX_SIZE);
            exit(EXIT_FAILURE);
        }
    }
    // check if EOF came from an end-of-file or an error
    if (ferror(fp)) {
        printf("EOF read error: could not open %s. %s.\n", fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    env->ysize = y;
    env->state = malloc(sizeof(double) * 8);
    xcsf->num_actions = 8;
    xcsf->num_x_vars = 8;
    xcsf->num_y_vars = 1;
    xcsf->env = env;
    fclose(fp);
}

/**
 * @brief Frees the maze environment.
 * @param xcsf The XCSF data structure.
 */
void env_maze_free(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    free(env->state);
    free(env);
}

/**
 * @brief Resets the animat to a random empty position in the maze.
 * @param xcsf The XCSF data structure.
 */
void env_maze_reset(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    env->reset = false;
    do {
        env->xpos = irand_uniform(0, env->xsize);
        env->ypos = irand_uniform(0, env->ysize);
    } while(env->maze[env->ypos][env->xpos] != '*');

    if(MAZE_DEBUG && !xcsf->train) {
        printf("------------\n");
        env_maze_print(xcsf);
    }
}

/**
 * @brief Returns whether the maze needs to be reset.
 * @param xcsf The XCSF data structure.
 * @return Whether the maze needs to be reset.
 */
_Bool env_maze_isreset(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    return env->reset;
}

/**
 * @brief Returns the current animat perceptions in the maze.
 * @param xcsf The XCSF data structure.
 * @return The current animat perceptions.
 */
double *env_maze_get_state(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    int spos = 0;
    for(int y = -1; y < 2; y++) {
        for(int x = -1; x < 2; x++) {
            // ignore current pos
            if(x == 0 && y == 0) {
                continue;
            }
            // toroidal maze
            int xsense = ((env->xpos + x) % env->xsize + env->xsize) % env->xsize;
            int ysense = ((env->ypos + y) % env->ysize + env->ysize) % env->ysize;
            char s = env->maze[ysense][xsense];
            // convert sensor to real number
            env->state[spos] = env_maze_sensor(xcsf, s);
            spos++;
        }
    }
    return env->state;
}

/**
 * @brief Returns a float encoding of a sensor perception.
 * @param xcsf The XCSF data structure.
 * @param s The char value of the sensor.
 * @return A float encoding of the sensor.
 */
double env_maze_sensor(XCSF *xcsf, char s)
{
    (void)xcsf;
    double ret = 0;
    switch(s) {
        case '*': ret = 0.1; break;
        case 'O': ret = 0.3; break;
        case 'G': ret = 0.5; break;
        case 'F': ret = 0.7; break;
        case 'Q': ret = 0.9; break;
        default:
            printf("unsupported maze state: %c\n", s);
            exit(EXIT_FAILURE);
    }
    return ret;
}

/**
 * @brief Executes the specified action and returns the payoff.
 * @param xcsf The XCSF data structure.
 * @param action The action to perform.
 * @return The payoff from performing the action.
 */
double env_maze_execute(XCSF *xcsf, int action)
{
    if(action < 0 || action > 7) {
        printf("invalid maze action\n");
        exit(EXIT_FAILURE);
    }
    ENV_MAZE *env = xcsf->env;
    // toroidal maze
    int newx = ((env->xpos + x_moves[action]) % env->xsize + env->xsize) % env->xsize;
    int newy = ((env->ypos + y_moves[action]) % env->ysize + env->ysize) % env->ysize;
    // make the move and recieve reward
    double reward = 0;
    switch(env->maze[newy][newx]) {
        case 'O':
        case 'Q':
            break;
        case '*':
            env->ypos = newy;
            env->xpos = newx;
            break;
        case 'F': 
        case 'G':
            env->ypos = newy;
            env->xpos = newx;
            env->reset = true;
            reward = MAX_PAYOFF;
            break;
        default:
            printf("invalid maze type\n");
            exit(EXIT_FAILURE);
    }
    if(MAZE_DEBUG && !xcsf->train) {
        env_maze_print(xcsf);
    }
    return reward;
}

/**
 * @brief Returns the maximum payoff value possible in the maze.
 * @param xcsf The XCSF data structure.
 * @return The maximum payoff.
 */
double env_maze_maxpayoff(XCSF *xcsf)
{
    (void)xcsf;
    return MAX_PAYOFF;
}

/**
 * @brief Returns whether the environment is a multistep problem.
 * @param xcsf The XCSF data structure.
 * @return True
 */
_Bool env_maze_multistep(XCSF *xcsf)
{
    (void)xcsf;
    return true;
}

/**
 * @brief Prints the current state of the maze environment.
 * @param xcsf The XCSF data structure.
 */
void env_maze_print(XCSF *xcsf)
{
    ENV_MAZE *env = xcsf->env;
    for(int y = 0; y < env->ysize; y++) {
        for(int x = 0; x < env->xsize; x++) {
            if(x == env->xpos && y == env->ypos) {
                printf("X");
            }
            else {
                printf("%c", env->maze[y][x]);
            }
        }
        printf("\n");
    }
    printf("\n");
}
