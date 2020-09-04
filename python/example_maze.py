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

"""This example demonstrates the XCSF multi-step reinforcement learning
mechanisms to solve discrete mazes loaded from a specified input file.
Classifiers are composed of ternary conditions, linear recursive least squares
predictions and integer actions."""

import os
import sys
import random
from turtle import Screen, Turtle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xcsf.xcsf as xcsf

class Maze:
    """Maze problem environment."""

    def __init__(self, file_name):
        """Constructs a new maze problem from the maze file_name."""
        self.maze = [] #: maze as read from the input file
        line = []
        path = os.path.normpath("../env/maze/"+file_name+".txt")
        with open(path) as f:
            while True:
                c = f.read(1)
                if not c:
                    break
                if c == '\n':
                    self.maze.append(line)
                    line = []
                else:
                    line.append(c)
        self.x_size = len(self.maze[0]) #: maze width
        self.y_size = len(self.maze) #: maze height
        self.state = np.zeros(8) #: current maze state
        self.x_pos = 0 #: current x position within the maze
        self.y_pos = 0 #: current y position within the maze
        self.is_reset = False #: whether the goal state has been reached
        self.max_payoff = 1 #: reward for finding the goal

    def reset(self):
        """Resets a maze problem: generating a new random start position."""
        self.is_reset = False
        while True:
            self.x_pos = random.randint(0, self.x_size - 1)
            self.y_pos = random.randint(0, self.y_size - 1)
            if self.maze[self.y_pos][self.x_pos] == '*':
                break

    def sensor(self, x_pos, y_pos):
        """Returns the real-number representation of a discrete maze cell."""
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
        print("invalid maze state: "+str(s))
        sys.exit()

    def update_state(self):
        """Sets the current state to a real-vector representing the sensory input."""
        spos = 0
        for y in range(-1, 2):
            for x in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                x_sense = ((self.x_pos + x) % self.x_size + self.x_size) % self.x_size
                y_sense = ((self.y_pos + y) % self.y_size + self.y_size) % self.y_size
                self.state[spos] = self.sensor(x_sense, y_sense)
                spos += 1

    def execute(self, act):
        """Executes an action within the maze and returns the immediate reward."""
        if act < 0 or act > 7:
            print("invalid maze action")
            sys.exit()
        x_moves = [0, 1, 1, 1, 0, -1, -1, -1]
        y_moves = [-1, -1, 0, 1, 1, 1, 0, -1]
        x_new = ((self.x_pos + x_moves[act]) % self.x_size + self.x_size) % self.x_size
        y_new = ((self.y_pos + y_moves[act]) % self.y_size + self.y_size) % self.y_size
        s = self.maze[y_new][x_new]
        if s in ('O', 'Q'):
            return 0
        self.x_pos = x_new
        self.y_pos = y_new
        if s == '*':
            return 0
        if s in ('F', 'G'):
            self.is_reset = True
            return self.max_payoff
        print("invalid maze type")
        sys.exit()

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
xcs.COND_TYPE = 6 # ternary conditions
xcs.COND_BITS = 2 # bits per maze sensor
xcs.PRED_TYPE = 3 # linear recursive least squares
xcs.ACT_TYPE = 0 # integers
xcs.BETA = 0.2 # classifier parameter update rate
xcs.THETA_EA = 25 # EA frequency
xcs.ALPHA = 0.1 # accuracy offset
xcs.NU = 5 # accuracy slope
xcs.EA_SUBSUMPTION = True
xcs.SET_SUBSUMPTION = True
xcs.THETA_SUB = 100 # minimum experience of a subsumer

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
bar = tqdm(total=N) # progress bar

MAZE_NAME = "maze4"
GOAL_STEPS = 3.5
maze = Maze(MAZE_NAME)

for i in range(N):
    for j in range(xcs.PERF_TRIALS):
        # explore trial
        maze.reset()
        xcs.init_trial()
        for k in range(xcs.TELETRANSPORTATION):
            xcs.init_step()
            maze.update_state()
            action = xcs.decision(maze.state, True) # explore
            reward = maze.execute(action)
            xcs.update(reward, maze.is_reset)
            xcs.end_step()
            if maze.is_reset:
                break
        xcs.end_trial()
        # exploit trial
        err = 0
        cnt = 0
        maze.reset()
        xcs.init_trial()
        for k in range(xcs.TELETRANSPORTATION):
            xcs.init_step()
            maze.update_state()
            action = xcs.decision(maze.state, False) # exploit
            reward = maze.execute(action)
            xcs.update(reward, maze.is_reset)
            err += xcs.error(reward, maze.is_reset, maze.max_payoff)
            cnt += 1
            xcs.end_step()
            if maze.is_reset:
                break
        xcs.end_trial()
        steps[i] += cnt
        error[i] += err / float(cnt)
    steps[i] /= float(xcs.PERF_TRIALS)
    error[i] /= float(xcs.PERF_TRIALS)
    trials[i] = (i + 1) * xcs.PERF_TRIALS
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # update status
    status = ("trials=%d steps=%.5f error=%.5f psize=%d msize=%.1f" %
              (trials[i], steps[i], error[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

#################################
# Plot XCSF learning performance
#################################

plt.figure(figsize=(10, 6))
plt.plot(trials, steps)
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=GOAL_STEPS, xmin=0, xmax=1, linestyle='dashed', color='k')
plt.title(MAZE_NAME, fontsize=14)
plt.ylabel('Steps to Goal', fontsize=12)
plt.xlabel('Trials', fontsize=12)
plt.xlim([0, N * xcs.PERF_TRIALS])
plt.show()

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

def draw_maze(XOFF, YOFF):
    """Draws the background and outline of the current maze."""
    bg = Turtle(visible=False)
    screen.tracer(False)
    bg.penup()
    bg.shape("square")
    bg.shapesize(1, 1)
    for y in range(maze.y_size):
        for x in range(maze.x_size):
            s = maze.maze[y][x]
            if s == '*':
                bg.color("white")
            if s == 'O':
                bg.color("black")
            if s == 'G':
                bg.color("yellow")
            if s == 'F':
                bg.color("yellow")
            if s == 'Q':
                bg.color("brown")
            bg.goto(XOFF + x * CELL_SIZE, YOFF + y * CELL_SIZE)
            bg.stamp()
    XOFF = XOFF - CELL_SIZE / 2
    YOFF = YOFF - CELL_SIZE / 2
    bg.goto(XOFF, YOFF)
    bg.pensize(2)
    bg.color("black")
    bg.pendown()
    bg.goto(XOFF, YOFF + GRID_HEIGHT * CELL_SIZE)
    bg.goto(XOFF + GRID_WIDTH * CELL_SIZE, YOFF + GRID_HEIGHT * CELL_SIZE)
    bg.goto(XOFF + GRID_WIDTH * CELL_SIZE, YOFF)
    bg.goto(XOFF, YOFF)
    bg.penup()

def visualise(XOFF, YOFF):
    """Executes an XCSF exploit run through the maze and draws the path."""
    agent = Turtle(visible=True)
    agent.shape("turtle")
    agent.color("green")
    agent.speed("normal")
    agent.shapesize(0.5, 0.5)
    agent.pensize(2)
    maze.reset()
    xcs.init_trial()
    for k in range(xcs.TELETRANSPORTATION):
        xcs.init_step()
        maze.update_state()
        if k == 0:
            agent.penup()
            agent.goto(XOFF + maze.x_pos * CELL_SIZE, YOFF + maze.y_pos * CELL_SIZE)
            screen.tracer(True)
            agent.pendown()
        action = xcs.decision(maze.state, False)
        reward = maze.execute(action)
        agent.goto(XOFF + maze.x_pos * CELL_SIZE, YOFF + maze.y_pos * CELL_SIZE)
        xcs.update(reward, maze.is_reset)
        xcs.end_step()
        if maze.is_reset:
            break
    xcs.end_trial()

GRID_XOFF = (GRID_WIDTH * CELL_SIZE)
GRID_YOFF = (GRID_HEIGHT * CELL_SIZE)

# draw some runs through the maze
for i in range(8):
    for j in range(4):
        XOFF = i * (GRID_XOFF + CELL_SIZE)
        YOFF = j * (GRID_YOFF + CELL_SIZE)
        draw_maze(XOFF, YOFF)
        visualise(XOFF, YOFF)

input("Press enter to exit.")
