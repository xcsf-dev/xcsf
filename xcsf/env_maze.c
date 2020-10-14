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
 * @date 2015--2020.
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
 * Woods 102: 3.31 \n
 * Maze F1: 1.8 \n
 * Maze F2: 2.5 \n
 * Maze F3: 3.375 \n
 * Maze F4: 4.5
 */

#include "env_maze.h"
#include "param.h"
#include "utils.h"

#define MAX_PAYOFF (1.) //!< The payoff provided at a food position

/**
 * @brief Maze x-axis moves.
 */
static const int x_moves[] = { 0, +1, +1, +1, 0, -1, -1, -1 };

/**
 * @brief Maze y-axis moves.
 */
static const int y_moves[] = { -1, -1, 0, +1, +1, +1, 0, -1 };

/**
 * @brief Returns a float encoding of a sensor perception.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] s The char value of the sensor.
 * @return A float encoding of the sensor.
 */
static double
env_maze_sensor(const struct XCSF *xcsf, const char s)
{
    (void) xcsf;
    switch (s) {
        case '*':
            return 0.1;
        case 'O':
            return 0.3;
        case 'G':
            return 0.5;
        case 'F':
            return 0.7;
        case 'Q':
            return 0.9;
        default:
            printf("unsupported maze state: %c\n", s);
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Initialises a maze environment from a specified file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] filename The file name of the specified maze environment.
 */
void
env_maze_init(struct XCSF *xcsf, const char *filename)
{
    // open maze file
    FILE *fp = fopen(filename, "rt");
    if (fp == 0) {
        printf("could not open %s. %s.\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }
    // read maze
    struct EnvMaze *env = malloc(sizeof(struct EnvMaze));
    int x = 0;
    int y = 0;
    int c = 0;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            ++y;
            env->xsize = x;
            x = 0;
        } else {
            env->maze[y][x] = (char) c;
            ++x;
        }
        // check maximum maze size not exceeded
        if (x > MAX_SIZE || y > MAX_SIZE) {
            printf("Maze too big. Max size = [%d,%d]\n", MAX_SIZE, MAX_SIZE);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
    }
    // check if EOF came from an end-of-file or an error
    if (ferror(fp)) {
        printf("EOF error opening %s. %s.\n", filename, strerror(errno));
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    env->ysize = y;
    env->state = malloc(sizeof(double) * 8);
    xcsf->env = env;
    fclose(fp);
    param_init(xcsf, 8, 1, 8);
}

/**
 * @brief Frees the maze environment.
 * @param [in] xcsf The XCSF data structure.
 */
void
env_maze_free(const struct XCSF *xcsf)
{
    struct EnvMaze *env = xcsf->env;
    free(env->state);
    free(env);
}

/**
 * @brief Resets the animat to a random empty position in the maze.
 * @param [in] xcsf The XCSF data structure.
 */
void
env_maze_reset(const struct XCSF *xcsf)
{
    struct EnvMaze *env = xcsf->env;
    env->done = false;
    do {
        env->xpos = rand_uniform_int(0, env->xsize);
        env->ypos = rand_uniform_int(0, env->ysize);
    } while (env->maze[env->ypos][env->xpos] != '*');
}

/**
 * @brief Returns whether the maze is in a terminal state.
 * @param [in] xcsf The XCSF data structure.
 * @return Whether the maze is in a terminal state.
 */
bool
env_maze_is_done(const struct XCSF *xcsf)
{
    const struct EnvMaze *env = xcsf->env;
    return env->done;
}

/**
 * @brief Returns the current animat perceptions in the maze.
 * @param [in] xcsf The XCSF data structure.
 * @return The current animat perceptions.
 */
const double *
env_maze_get_state(const struct XCSF *xcsf)
{
    const struct EnvMaze *env = xcsf->env;
    int spos = 0;
    for (int y = -1; y < 2; ++y) {
        for (int x = -1; x < 2; ++x) {
            if (x == 0 && y == 0) { // ignore current pos
                continue;
            }
            // toroidal maze
            const int xsense =
                ((env->xpos + x) % env->xsize + env->xsize) % env->xsize;
            const int ysense =
                ((env->ypos + y) % env->ysize + env->ysize) % env->ysize;
            const char s = env->maze[ysense][xsense];
            // convert sensor to real number
            env->state[spos] = env_maze_sensor(xcsf, s);
            ++spos;
        }
    }
    return env->state;
}

/**
 * @brief Executes the specified action and returns the payoff.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] action The action to perform.
 * @return The payoff from performing the action.
 */
double
env_maze_execute(const struct XCSF *xcsf, const int action)
{
    if (action < 0 || action > 7) {
        printf("invalid maze action\n");
        exit(EXIT_FAILURE);
    }
    struct EnvMaze *env = xcsf->env;
    // toroidal maze
    const int newx =
        ((env->xpos + x_moves[action]) % env->xsize + env->xsize) % env->xsize;
    const int newy =
        ((env->ypos + y_moves[action]) % env->ysize + env->ysize) % env->ysize;
    // make the move and recieve reward
    double reward = 0;
    switch (env->maze[newy][newx]) {
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
            env->done = true;
            reward = MAX_PAYOFF;
            break;
        default:
            printf("invalid maze type\n");
            exit(EXIT_FAILURE);
    }
    return reward;
}

/**
 * @brief Returns the maximum payoff value possible in the maze.
 * @param [in] xcsf The XCSF data structure.
 * @return The maximum payoff.
 */
double
env_maze_maxpayoff(const struct XCSF *xcsf)
{
    (void) xcsf;
    return MAX_PAYOFF;
}

/**
 * @brief Returns whether the environment is a multistep problem.
 * @param [in] xcsf The XCSF data structure.
 * @return True
 */
bool
env_maze_multistep(const struct XCSF *xcsf)
{
    (void) xcsf;
    return true;
}
