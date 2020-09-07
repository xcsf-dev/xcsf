#!/usr/bin/python3
#
# Copyright (C) 2020 Richard Preen <rpreen@gmail.com>
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
to solve the cart-pole problem from the OpenAI Gym.
"""

import numpy as np
import gym
import xcsf.xcsf as xcsf

############################################
# Initialise OpenAI Gym problem environment
############################################

env = gym.make('CartPole-v0')
X_DIM = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
MAX_PAYOFF = 1

###################
# Initialise XCSF
###################

xcs = xcsf.XCS(X_DIM, 1, N_ACTIONS)

xcs.OMP_NUM_THREADS = 8
xcs.POP_SIZE = 500
xcs.PERF_TRIALS = 100
xcs.EPS_0 = 0.01 # target error
xcs.GAMMA = 0.9 # discount rate for delayed reward
xcs.BETA = 0.1 # classifier parameter update rate
xcs.ALPHA = 0.1 # accuracy offset
xcs.NU = 5 # accuracy slope
xcs.EA_SUBSUMPTION = False
xcs.SET_SUBSUMPTION = False
xcs.THETA_EA = 50 # EA invocation frequency

xcs.MAX_NEURON_GROW = 1 # max neurons to add/remove per mut
xcs.COND_TYPE = 3 # neural network conditions
xcs.COND_OUTPUT_ACTIVATION = 3 # linear
xcs.COND_HIDDEN_ACTIVATION = 9 # selu
xcs.COND_NUM_NEURONS = [10] # initial neurons
xcs.COND_MAX_NEURONS = [100] # maximum neurons
xcs.COND_EVOLVE_WEIGHTS = True
xcs.COND_EVOLVE_NEURONS = True
xcs.COND_EVOLVE_FUNCTIONS = False
xcs.COND_EVOLVE_CONNECTIVITY = False

xcs.ACT_TYPE = 0 # integer actions

xcs.PRED_TYPE = 5 # neural network predictions
xcs.PRED_OUTPUT_ACTIVATION = 3 # linear
xcs.PRED_HIDDEN_ACTIVATION = 9 # selu
xcs.PRED_NUM_NEURONS = [10] # initial neurons
xcs.PRED_MAX_NEURONS = [10] # maximum neurons
xcs.PRED_EVOLVE_WEIGHTS = True
xcs.PRED_EVOLVE_NEURONS = False
xcs.PRED_EVOLVE_FUNCTIONS = False
xcs.PRED_EVOLVE_CONNECTIVITY = False
xcs.PRED_EVOLVE_ETA = True
xcs.PRED_SGD_WEIGHTS = True
xcs.PRED_ETA = 0.001 # maximum gradient descent rate
xcs.PRED_DECAY = 0 # weight decay

xcs.print_params()

#####################
# Execute experiment
#####################

N = 40 # maximum 4000 learning trials/episodes
trials = np.zeros(N)
psize = np.zeros(N)
msize = np.zeros(N)
score = np.zeros(N)
error = np.zeros(N)

for i in range(N):
    for j in range(xcs.PERF_TRIALS):
        # explore trial/episode
        state = env.reset()
        xcs.init_trial()
        while True:
            xcs.init_step()
            action = xcs.decision(state, True)
            next_state, reward, is_reset, info = env.step(action)
            xcs.update(reward, is_reset)
            xcs.end_step()
            state = next_state
            if is_reset:
                break
        xcs.end_trial()
        # exploit trial/episode
        episode_score = 0
        cnt = 0
        err = 0
        state = env.reset()
        xcs.init_trial()
        while True:
            xcs.init_step()
            action = xcs.decision(state, False)
            next_state, reward, is_reset, info = env.step(action)
            xcs.update(reward, is_reset)
            err += xcs.error(reward, is_reset, MAX_PAYOFF)
            episode_score += reward
            cnt += 1
            xcs.end_step()
            state = next_state
            if is_reset:
                break
        xcs.end_trial()
        score[i] += episode_score
        error[i] += err / float(cnt)
    score[i] /= float(xcs.PERF_TRIALS)
    error[i] /= float(xcs.PERF_TRIALS)
    trials[i] = (i + 1) * xcs.PERF_TRIALS
    psize[i] = xcs.pop_size()
    msize[i] = xcs.msetsize()
    status = ("episodes=%d score=%.2f error=%.5f psize=%d msize=%.1f" %
              (trials[i], score[i], error[i], psize[i], msize[i]))
    print(status)
    if score[i] > env.spec.reward_threshold: # solved
        print("solved: score %.2f > %.2f" % (score[i], env.spec.reward_threshold))
        break

# display 10 exploit trials/episodes
N = 10
for i in range(N):
    episode_score = 0
    err = 0
    cnt = 0
    state = env.reset()
    xcs.init_trial()
    while True:
        env.render()
        xcs.init_step()
        action = xcs.decision(state, False)
        next_state, reward, is_reset, info = env.step(action)
        xcs.update(reward, is_reset)
        err += xcs.error(reward, is_reset, MAX_PAYOFF)
        cnt += 1
        episode_score += reward
        xcs.end_step()
        state = next_state
        if is_reset:
            break
    xcs.end_trial()
    perf = (episode_score / env._max_episode_steps) * 100
    print("exploit %d/%d: perf=%.2f%%, score=%.2f, error=%.5f" %
          (i+1, N, perf, episode_score, err / cnt))
env.close()
