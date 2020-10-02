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
This example demonstrates XCSF (single-step) reinforcement learning applied to
the real-multiplexer problem. Classifiers are composed of hyperrectangle
conditions, linear least squares predictions, and integer actions.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xcsf.xcsf as xcsf

class Mux:
    """
    Real-multiplexer problem environment.

    The Mux class generates random real vectors of length k+pow(2,k) where the
    first k bits determine the position of the output bit in the last pow(2,k)
    bits. E.g., for a 3-bit problem, the first (rounded) bit addresses which of the
    following 2 bits are the (rounded) output.

    Example valid lengths: 3, 6, 11, 20, 37, 70, 135, 264.
    """

    def __init__(self, n_bits):
        """ Constructs a new real-multiplexer problem of maximum size n_bits. """
        self.n_bits = n_bits #: total number of bits
        self.n_actions = 2 #: total number of actions
        self.state = np.zeros(n_bits) #: current mux state
        self.max_payoff = 1 #: reward for a correct prediction
        self.pos_bits = 1 #: number of addressing bits
        while self.pos_bits + pow(2, self.pos_bits) <= self.n_bits:
            self.pos_bits += 1
        self.pos_bits -= 1
        print(str(self.n_bits)+" bits, "+str(self.pos_bits)+" position bits")

    def reset(self):
        """ Generates a random real-multiplexer state. """
        for k in range(self.n_bits):
            self.state[k] = random.random()

    def answer(self):
        """ Returns the (discretised) bit addressed by the current mux state. """
        pos = self.pos_bits
        for k in range(self.pos_bits):
            if self.state[k] > 0.5:
                pos += pow(2, self.pos_bits - 1 - k)
        if self.state[pos] > 0.5:
            return 1
        return 0

    def execute(self, act):
        """ Returns the reward for performing an action. """
        if act == self.answer():
            return self.max_payoff
        return 0

# Create new real-multiplexer problem
mux = Mux(6)
X_DIM = mux.n_bits
N_ACTIONS = mux.n_actions
MAX_PAYOFF = mux.max_payoff

###################
# Initialise XCSF
###################

# constructor = (x_dim, y_dim, n_actions)
xcs = xcsf.XCS(X_DIM, 1, N_ACTIONS)

# override default.ini
xcs.OMP_NUM_THREADS = 8 # number of CPU cores to use
xcs.POP_SIZE = 1000 # maximum population size
xcs.EPS_0 = 0.01 # target error
xcs.COND_TYPE = 1 # hyperrectangles
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

PERF_TRIALS = 1000 # number of trials over which to average performance
N = 100 # 100,000 trials in total to run
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
performance = np.zeros(N)
error = np.zeros(N)

def egreedy_action(state, epsilon):
    """ Selects an action using an epsilon greedy policy. """
    if np.random.rand() < epsilon:
        return random.randrange(N_ACTIONS), 0
    prediction_array = xcs.predict(state.reshape(1,-1))[0]
    action = np.argmax(prediction_array)
    prediction = prediction_array[action]
    return action, prediction

def run_experiment():
    """ Executes a single experiment. """
    bar = tqdm(total=N) # progress bar
    for i in range(N):
        for _ in range(PERF_TRIALS):
            # explore trial
            mux.reset()
            action, prediction = egreedy_action(mux.state, 1) # random action
            reward = mux.execute(action)
            xcs.fit(mux.state, action, reward) # update action set, run EA, etc.
            # exploit trial
            mux.reset()
            action, prediction = egreedy_action(mux.state, 0) # best action
            reward = mux.execute(action)
            performance[i] += reward / MAX_PAYOFF
            error[i] += abs(reward - prediction) / MAX_PAYOFF
        performance[i] /= float(PERF_TRIALS)
        error[i] /= PERF_TRIALS
        trials[i] = xcs.time() # number of learning updates performed
        psize[i] = xcs.pop_size() # current population size
        msize[i] = xcs.msetsize() # avg match set size
        # update status
        status = ('trials=%d performance=%.5f error=%.5f psize=%d msize=%.1f' %
                (trials[i], performance[i], error[i], psize[i], msize[i]))
        bar.set_description(status)
        bar.refresh()
        bar.update(1)
    bar.close()

# run
run_experiment()

#################################
# Plot XCSF learning performance
#################################

plt.figure(figsize=(10, 6))
plt.plot(trials, performance, label='Performance')
plt.plot(trials, error, label='System error')
plt.grid(linestyle='dotted', linewidth=1)
plt.title(str(mux.n_bits)+'-bit Real Multiplexer', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.xlim([0, N * PERF_TRIALS])
plt.legend()
plt.show()
