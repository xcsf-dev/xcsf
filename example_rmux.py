#!/usr/bin/python3
#
# Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
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

import xcsf.xcsf as xcsf # Import XCSF
import numpy as np
from random import random
import matplotlib.pyplot as plt
from tqdm import tqdm
np.set_printoptions(suppress=True)

###########################
# Real-multiplexer problem
###########################

mux = 6 # total number of bits

# set the number of position bits
pos_bits = 1
while pos_bits + pow(2, pos_bits) <= mux:
    pos_bits += 1
pos_bits -= 1

print(str(mux)+" bits, "+str(pos_bits)+" position bits")

# current mux state
state = np.zeros(mux)

# random mux
def rmux_reset():
    for i in range(mux):
        state[i] = random()

# calculate mux answer
def rmux_answer(state):
    pos = pos_bits
    for i in range(pos_bits):
        if state[i] > 0.5:
            pos += pow(2, pos_bits - 1 - i)
    if state[pos] > 0.5:
        return 1
    else:
        return 0

###################
# Initialise XCSF
###################

# initialise XCSF for single-step reinforcement learning
xcs = xcsf.XCS(mux, 2, False)

# override default.ini
xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 1000
xcs.PERF_AVG_TRIALS = 1000
xcs.EPS_0 = 0.01 # target error
xcs.COND_TYPE = 0 # hyperrectangles
xcs.PRED_TYPE = 0 # linear least squares
xcs.ACT_TYPE = 0 # integers

##################################
# Example plotting in matplotlib
##################################

n = 100 # 100,000 trials
trials = np.zeros(n)
psize = np.zeros(n)
msize = np.zeros(n)
performance = np.zeros(n)
error = np.zeros(n)
bar = tqdm(total=n) # progress bar

for i in range(n):
    perf = 0
    err = 0
    for j in range(xcs.PERF_AVG_TRIALS):
        # new problem
        rmux_reset()
        answer = rmux_answer(state)
        # explore trial
        xcs.single_reset()
        action = xcs.single_decision(state, True)
        if action == answer:
            reward = 1
        else:
            reward = 0
        xcs.single_update(reward)
        # new problem
        rmux_reset()
        answer = rmux_answer(state)
        # exploit trial
        xcs.single_reset()
        action = xcs.single_decision(state, False)
        if action == answer:
            reward = 1
        else:
            reward = 0
        perf += reward
        err += xcs.single_error(reward)
    performance[i] = perf / xcs.PERF_AVG_TRIALS
    error[i] = err / xcs.PERF_AVG_TRIALS
    trials[i] = xcs.time() # number of trials so far
    psize[i] = xcs.pop_size() # current population size
    msize[i] = xcs.msetsize() # avg match set size
    # update status
    status = ("trials=%d performance=%.5f error=%.5f psize=%d msize=%.1f smut=%.4f pmut=%.4f emut=%.4f fmut=%.4f"
        % (trials[i], performance[i], error[i], psize[i], msize[i], xcs.pop_avg_mu(0), xcs.pop_avg_mu(1), xcs.pop_avg_mu(2), xcs.pop_avg_mu(3)))
    bar.set_description(status)
    bar.refresh()
    bar.update(1)
bar.close()

# plot XCSF learning performance
plt.figure(figsize=(10,6))
plt.plot(trials, performance, label='Performance')
plt.plot(trials, error, label='System error')
plt.grid(linestyle='dotted', linewidth=1)
plt.title('XCSF Training Performance', fontsize=14)
plt.xlabel('Trials', fontsize=12)
plt.xlim([0,n*xcs.PERF_AVG_TRIALS])
plt.legend()
plt.show()
