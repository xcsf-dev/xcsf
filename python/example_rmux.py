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

"""This example demonstrates the XCSF reinforcement learning mechanisms applied
to the real-multiplexer problem. Classifiers are composed of hyperrectangle
conditions, linear least squares predictions, and integer actions."""

from random import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xcsf.xcsf as xcsf

class Mux:
    """Real-multiplexer problem environment."""

    def __init__(self, n_bits):
        """Constructs a new real-multiplexer problem of maximum size n_bits."""
        self.n_bits = n_bits #: total number of bits
        self.n_actions = 2 #: total number of actions
        self.state = np.zeros(n_bits) #: current mux state
        self.is_reset = True #: single-step problem: always reset the trial
        self.max_payoff = 1 #: reward for a correct prediction
        self.pos_bits = 1 #: number of addressing bits
        while self.pos_bits + pow(2, self.pos_bits) <= self.n_bits:
            self.pos_bits += 1
        self.pos_bits -= 1
        print(str(self.n_bits)+" bits, "+str(self.pos_bits)+" position bits")

    def reset(self):
        """Generates a random real-multiplexer state."""
        for k in range(self.n_bits):
            self.state[k] = random()

    def answer(self):
        """Returns the (discretised) bit addressed by the current mux state."""
        pos = self.pos_bits
        for k in range(self.pos_bits):
            if self.state[k] > 0.5:
                pos += pow(2, self.pos_bits - 1 - k)
        if self.state[pos] > 0.5:
            return 1
        return 0

    def execute(self, act):
        """Returns the reward for performing an action."""
        if act == self.answer():
            return self.max_payoff
        return 0

# Create new real-multiplexer problem
mux = Mux(6)

###################
# Initialise XCSF
###################

# initialise XCSF for reinforcement learning
xcs = xcsf.XCS(mux.n_bits, 1, mux.n_actions)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 1000
xcs.PERF_TRIALS = 1000
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

N = 100 # 100,000 trials
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
performance = np.zeros(N)
error = np.zeros(N)
bar = tqdm(total=N) # progress bar

for i in range(N):
    for j in range(xcs.PERF_TRIALS):
        # explore trial
        mux.reset()
        xcs.init_trial()
        xcs.init_step()
        action = xcs.decision(mux.state, True) # explore
        reward = mux.execute(action)
        xcs.update(reward, mux.is_reset)
        xcs.end_step()
        xcs.end_trial()
        # exploit trial
        mux.reset()
        xcs.init_trial()
        xcs.init_step()
        action = xcs.decision(mux.state, False) # exploit
        reward = mux.execute(action)
        performance[i] += reward
        error[i] += xcs.error(reward, mux.is_reset, mux.max_payoff)
        xcs.end_step()
        xcs.end_trial()
    performance[i] /= float(xcs.PERF_TRIALS)
    error[i] /= float(xcs.PERF_TRIALS)
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # update status
    status = ("trials=%d performance=%.5f error=%.5f psize=%d msize=%.1f" %
              (trials[i], performance[i], error[i], psize[i], msize[i]))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

#################################
# Plot XCSF learning performance
#################################

plt.figure(figsize=(10, 6))
plt.plot(trials, performance, label='Performance')
plt.plot(trials, error, label='System error')
plt.grid(linestyle='dotted', linewidth=1)
plt.title(str(mux.n_bits)+'-bit Real Multiplexer', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.xlim([0, N * xcs.PERF_TRIALS])
plt.legend()
plt.show()
