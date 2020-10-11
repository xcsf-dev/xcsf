#!/usr/bin/python3
#
# Copyright (C) 2019--2020 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
This example demonstrates the XCSF multi-step reinforcement learning mechanisms
to solve discrete mazes loaded from a specified input file.
"""

import os
import sys
import random
from turtle import Screen, Turtle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xcsf.xcsf as xcsf

class Maze:
    """
    Maze problem environment.

    The maze class reads in the chosen maze from a file where each entry
    specifies a distinct position in the maze. The maze is toroidal and if the
    agent/animat reaches one edge it can reenter the maze from the other side.
    Obstacles are coded as 'O' and 'Q', empty positions as '*', and food as 'F'
    or 'G'. The 8 adjacent cells are perceived by the animat and 8 movements
    are possible to one of the adjacent cells (if not blocked.) The animat is
    initially placed at a random empty position. The goal is to find the
    shortest path to the food.

    Some mazes require a form of memory to be solved optimally.
    """

    OPTIMAL = { 'woods1': 1.7, 'woods2': 1.7, 'woods14': 9.5, 'maze4': 3.5,
            'maze5': 4.61, 'maze6': 5.19, 'maze7': 4.33, 'maze10': 5.11,
            'woods101': 2.9, 'woods101half': 3.1, 'woods102': 3.31,
            'mazef1': 1.8, 'mazef2': 2.5, 'mazef3': 3.375, 'mazef4': 4.5 }
    MAX_PAYOFF = 1 #: reward for finding the goal
    X_MOVES = [0, 1, 1, 1, 0, -1, -1, -1] #: possible moves on x-axis
    Y_MOVES = [-1, -1, 0, 1, 1, 1, 0, -1] #: possible moves on y-axis

    def __init__(self, filename):
        """ Constructs a new maze problem given a maze file name. """
        self.name = filename #: maze name
        self.maze = [] #: maze as read from the input file
        line = []
        path = os.path.normpath('../env/maze/'+filename+'.txt')
        with open(path) as f:
            while True:
                c = f.read(1)
                if not c:
                    break
                if c == '\n':
                    self.maze.insert(0, line)
                    line = []
                else:
                    line.append(c)
        self.x_size = len(self.maze[0]) #: maze width
        self.y_size = len(self.maze) #: maze height
        self.state = np.zeros(8) #: current maze state
        self.x_pos = 0 #: current x position within the maze
        self.y_pos = 0 #: current y position within the maze

    def reset(self):
        """ Resets a maze problem: generating a new random start position. """
        while True:
            self.x_pos = random.randint(0, self.x_size - 1)
            self.y_pos = random.randint(0, self.y_size - 1)
            if self.maze[self.y_pos][self.x_pos] == '*':
                break
        self.update_state()
        return np.copy(self.state)

    def sensor(self, x_pos, y_pos):
        """ Returns the real-number representation of a discrete maze cell. """
        s = self.maze[y_pos][x_pos]
        if s == '*':
            return 0.1
        if s == 'O':
            return 0.3
        if s == 'G':
            return 0.5
        if s == 'F':
            return 0.7
        if s == 'Q':
            return 0.9
        print('invalid maze state: '+str(s))
        sys.exit()

    def update_state(self):
        """ Sets the state to a real-vector representing the sensory input. """
        spos = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if j == 0 and i == 0:
                    continue
                x = ((self.x_pos + j) % self.x_size + self.x_size) % self.x_size
                y = ((self.y_pos + i) % self.y_size + self.y_size) % self.y_size
                self.state[spos] = self.sensor(x, y)
                spos += 1

    def step(self, act):
        """
        Takes a step in the maze, performing the specified action.
        Returns next state, immediate reward and whether terminal state reached.
        """
        if act < 0 or act > 7:
            print('invalid maze action')
            sys.exit()
        x_vec = Maze.X_MOVES[act]
        y_vec = Maze.Y_MOVES[act]
        x_new = ((self.x_pos + x_vec) % self.x_size + self.x_size) % self.x_size
        y_new = ((self.y_pos + y_vec) % self.y_size + self.y_size) % self.y_size
        s = self.maze[y_new][x_new]
        if s in ('O', 'Q'):
            return np.copy(self.state), 0, False
        self.x_pos = x_new
        self.y_pos = y_new
        self.update_state()
        if s == '*':
            return np.copy(self.state), 0, False
        if s in ('F', 'G'):
            return np.copy(self.state), self.max_payoff(), True
        print('invalid maze type')
        sys.exit()

    def optimal(self):
        """ Returns the optimal number of steps to the goal. """
        return Maze.OPTIMAL[self.name]

    def max_payoff(self):
        """ Returns the reward for reaching the goal state. """
        return float(Maze.MAX_PAYOFF)

###################
# Initialise XCSF
###################

# initialise XCSF for reinforcement learning
X_DIM = 8
Y_DIM = 1
N_ACTIONS = 8
xcs = xcsf.XCS(X_DIM, Y_DIM, N_ACTIONS)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 1000
xcs.PERF_TRIALS = 50
xcs.EPS_0 = 0.001 # target error
xcs.BETA = 0.2 # classifier parameter update rate
xcs.THETA_EA = 25 # EA frequency
xcs.ALPHA = 0.1 # accuracy offset
xcs.NU = 5 # accuracy slope
xcs.EA_SUBSUMPTION = True
xcs.SET_SUBSUMPTION = True
xcs.THETA_SUB = 100 # minimum experience of a subsumer
xcs.action('integer') # integer actions
xcs.condition('ternary', { 'bits': 2 }) # ternary conditions: 2-bits per float
xcs.prediction('rls-linear') # linear recursive least squares predictions

xcs.print_params()

#####################
# Execute experiment
#####################

N = 40 # 2,000 trials
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
steps = np.zeros(N)
error = np.zeros(N)

def trial(env, explore):
    """ Executes a single trial/episode. """
    err = 0
    state = env.reset()
    xcs.init_trial()
    for cnt in range(xcs.TELETRANSPORTATION):
        xcs.init_step()
        action = xcs.decision(state, explore)
        next_state, reward, done = env.step(action)
        xcs.update(reward, done)
        err += xcs.error(reward, done, env.max_payoff())
        xcs.end_step()
        if done:
            break
        state = next_state
    cnt += 1
    xcs.end_trial()
    return cnt, err / cnt

def run_experiment(env):
    """ Executes a single experiment. """
    bar = tqdm(total=N) # progress bar
    for i in range(N):
        for _ in range(xcs.PERF_TRIALS):
            # explore
            trial(env, True)
            # exploit
            cnt, err = trial(env, False)
            steps[i] += cnt
            error[i] += err
        steps[i] /= float(xcs.PERF_TRIALS)
        error[i] /= float(xcs.PERF_TRIALS)
        trials[i] = (i + 1) * xcs.PERF_TRIALS
        psize[i] = xcs.pop_size() # current population size
        msize[i] = xcs.msetsize() # avg match set size
        # update status
        status = ('trials=%d steps=%.5f error=%.5f psize=%d msize=%.1f' %
                (trials[i], steps[i], error[i], psize[i], msize[i]))
        bar.set_description(status)
        bar.refresh()
        bar.update(1)
    bar.close()

def plot_performance(env):
    """ Plots learning performance. """
    plt.figure(figsize=(10, 6))
    plt.plot(trials, steps)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.axhline(y=env.optimal(), xmin=0, xmax=1, linestyle='--', color='k')
    plt.title(env.name, fontsize=14)
    plt.ylabel('Steps to Goal', fontsize=12)
    plt.xlabel('Trials', fontsize=12)
    plt.xlim([0, N * xcs.PERF_TRIALS])
    plt.show()

maze = Maze('maze4')
run_experiment(maze)
plot_performance(maze)

#################################
# Visualise some maze runs
#################################

GRID_WIDTH = maze.x_size
GRID_HEIGHT = maze.y_size
CELL_SIZE = 20
WIDTH, HEIGHT = 1400, 720
screen = Screen()
screen.setup(WIDTH + 4, HEIGHT + 8)
screen.setworldcoordinates(0, 0, WIDTH, HEIGHT)

def draw_maze(xoff, yoff):
    """ Draws the background and outline of the current maze. """
    bg = Turtle(visible=False)
    screen.tracer(False)
    bg.penup()
    bg.shape('square')
    bg.shapesize(1, 1)
    for y in range(maze.y_size):
        for x in range(maze.x_size):
            s = maze.maze[y][x]
            if s == '*':
                bg.color('white')
            if s == 'O':
                bg.color('black')
            if s == 'G':
                bg.color('yellow')
            if s == 'F':
                bg.color('yellow')
            if s == 'Q':
                bg.color('brown')
            bg.goto(xoff + x * CELL_SIZE, yoff + y * CELL_SIZE)
            bg.stamp()
    xoff = xoff - CELL_SIZE / 2
    yoff = yoff - CELL_SIZE / 2
    bg.goto(xoff, yoff)
    bg.pensize(2)
    bg.color('black')
    bg.pendown()
    bg.goto(xoff, yoff + GRID_HEIGHT * CELL_SIZE)
    bg.goto(xoff + GRID_WIDTH * CELL_SIZE, yoff + GRID_HEIGHT * CELL_SIZE)
    bg.goto(xoff + GRID_WIDTH * CELL_SIZE, yoff)
    bg.goto(xoff, yoff)
    bg.penup()

def visualise(xoff, yoff):
    """ Executes an XCSF exploit run through the maze and draws the path. """
    state = maze.reset()
    agent = Turtle(visible=True)
    agent.shape('turtle')
    agent.color('green')
    agent.speed('normal')
    agent.shapesize(0.5, 0.5)
    agent.pensize(2)
    agent.penup()
    agent.goto(xoff + maze.x_pos * CELL_SIZE, yoff + maze.y_pos * CELL_SIZE)
    agent.pendown()
    screen.tracer(True)
    xcs.init_trial()
    for _ in range(xcs.TELETRANSPORTATION):
        xcs.init_step()
        action = xcs.decision(state, False)
        next_state, reward, done = maze.step(action)
        agent.goto(xoff + maze.x_pos * CELL_SIZE, yoff + maze.y_pos * CELL_SIZE)
        xcs.update(reward, done)
        xcs.end_step()
        if done:
            break
        state = next_state
    xcs.end_trial()

def draw_runs():
    """ Draw some runs through the maze. """
    grid_xoff = (GRID_WIDTH * CELL_SIZE)
    grid_yoff = (GRID_HEIGHT * CELL_SIZE)
    for i in range(8):
        for j in range(4):
            xoff = i * (grid_xoff + CELL_SIZE)
            yoff = j * (grid_yoff + CELL_SIZE)
            draw_maze(xoff, yoff)
            visualise(xoff, yoff)

draw_runs()
input('Press enter to exit.')
