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

################################################################################
# This example uses the multi-step reinforcement learning mechanism to solve
# mazes loaded from an input file.
################################################################################

import xcsf.xcsf as xcsf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

###########################
# Maze Environment
###########################

class Maze:

    def __init__(self, filename):
        self.maze = []
        line = []
        path = os.path.normpath("../env/maze/"+filename+".txt")
        with open(path) as f:
            while True:
                c = f.read(1)
                if not c:
                    break
                elif c == '\n':
                    self.maze.append(line)
                    line = []
                else:
                    line.append(c)
        self.x_size = len(self.maze[0])
        self.y_size = len(self.maze)
        self.state = np.zeros(8)
        self.x_pos = 0
        self.y_pos = 0
        self.is_reset = False
        self.max_payoff = 1

    def reset(self):
        self.is_reset = False
        while True:
            self.x_pos = random.randint(0, self.x_size-1)
            self.y_pos = random.randint(0, self.y_size-1)
            if self.maze[self.y_pos][self.x_pos] == '*':
                break

    def sensor(self, s):
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
        exit()
        
    def update_state(self):
        spos = 0
        for y in range(-1,2):
            for x in range(-1,2):
                if x == 0 and y == 0:
                    continue
                x_sense = ((self.x_pos + x) % self.x_size + self.x_size) % self.x_size
                y_sense = ((self.y_pos + y) % self.y_size + self.y_size) % self.y_size
                s = self.maze[y_sense][x_sense]
                self.state[spos] = self.sensor(s)
                spos += 1

    def execute(self, action):
        if action < 0 or action > 7:
            print("invalid maze action")
            exit()
        x_moves = [ 0, +1, +1, +1,  0, -1, -1, -1]
        y_moves = [-1, -1,  0, +1, +1, +1,  0, -1]
        x_new = ((self.x_pos + x_moves[action]) % self.x_size + self.x_size) % self.x_size
        y_new = ((self.y_pos + y_moves[action]) % self.y_size + self.y_size) % self.y_size
        s = self.maze[y_new][x_new]
        if s == 'O' or s == 'Q':
            return 0
        self.x_pos = x_new
        self.y_pos = y_new
        if s == '*':
            return 0
        if s == 'F' or s == 'G':
            self.is_reset = True
            return self.max_payoff
        print("invalid maze type")
        exit()

###################
# Initialise XCSF
###################

# initialise XCSF for multi-step reinforcement learning
x_dim = 8
n_actions = 8
xcs = xcsf.XCS(x_dim, n_actions, True)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 2000
xcs.PERF_TRIALS = 50
xcs.EPS_0 = 0.01 # target error
xcs.COND_TYPE = 6 # ternary conditions
xcs.COND_BITS = 2 # bits per maze sensor
xcs.PRED_TYPE = 1 # linear least squares
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

n = 40 # 2,000 trials
trials = np.zeros(n)
psize = np.zeros(n)
msize = np.zeros(n)
steps = np.zeros(n)
error = np.zeros(n)
bar = tqdm(total=n) # progress bar

MAZE_NAME = "maze4"
goal_steps = 3.5
maze = Maze(MAZE_NAME)

for i in range(n):
    for j in range(xcs.PERF_TRIALS):
        # explore trial
        maze.reset()
        xcs.multi_init_trial()
        for k in range(xcs.TELETRANSPORTATION):
            xcs.multi_init_step()
            maze.update_state()
            action = xcs.multi_decision(maze.state, True)
            reward = maze.execute(action)
            xcs.multi_update(reward, maze.is_reset)
            xcs.multi_end_step()
            if maze.is_reset:
                break
        xcs.multi_end_trial()
        # exploit trial
        err = 0
        cnt = 0
        maze.reset()
        xcs.multi_init_trial()
        for k in range(xcs.TELETRANSPORTATION):
            xcs.multi_init_step()
            maze.update_state()
            action = xcs.multi_decision(maze.state, False)
            reward = maze.execute(action)
            xcs.multi_update(reward, maze.is_reset)
            err += xcs.multi_error(reward, maze.is_reset, maze.max_payoff)
            cnt += 1
            xcs.multi_end_step()
            if maze.is_reset:
                break
        xcs.multi_end_trial()
        steps[i] += cnt
        error[i] += err / float(cnt)
    steps[i] /= float(xcs.PERF_TRIALS)
    error[i] /= float(xcs.PERF_TRIALS)
    trials[i] = (i+1) * xcs.PERF_TRIALS
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # update status
    status = ("trials=%d steps=%.5f error=%.5f psize=%d msize=%.1f"
        % (trials[i], steps[i], error[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

#################################
# Plot XCSF learning performance
#################################

plt.figure(figsize=(10,6))
plt.plot(trials, steps)
plt.grid(linestyle='dotted', linewidth=1)
plt.axhline(y=goal_steps, xmin=0, xmax=1, linestyle='dashed', color='k')
plt.title(MAZE_NAME, fontsize=14)
plt.ylabel('Steps to Goal', fontsize=12)
plt.xlabel('Trials', fontsize=12)
plt.xlim([0, n * xcs.PERF_TRIALS])
plt.show()
