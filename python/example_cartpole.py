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

import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation
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
SAVE_GIF = False

def save_frames_as_gif(frames, fscore, path='./', filename='animation.gif'):
    """Save animation as gif"""
    rcParams['font.family'] = 'monospace'
    fig = plt.figure(dpi=90)
    fig.set_size_inches(3, 3)
    ax = fig.add_subplot(111)
    patch = plt.imshow(frames[0])
    bbox = dict(boxstyle="round", fc="0.8")
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
        strial = str(ftrial[i])
        sscore = str(int(fscore[i]))
        text = ('trial = %5s, score = %3s' % (strial, sscore))
        ax.annotate(text, xy=(0,100), xytext=(-30,1), fontsize=12, bbox=bbox)
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
            interval=100, blit=False)
    anim.save(path + filename, writer='imagemagick', fps=30)

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
xcs.THETA_EA = 100 # EA invocation frequency
xcs.THETA_DEL = 100 # min experience before fitness used for deletion
xcs.P_EXPLORE = 1 # probability of picking a random action during exploration

xcs.MAX_NEURON_GROW = 1 # max neurons to add/remove per mut
xcs.COND_TYPE = 3 # neural network conditions
xcs.COND_OUTPUT_ACTIVATION = 3 # linear
xcs.COND_HIDDEN_ACTIVATION = 9 # selu
xcs.COND_NUM_NEURONS = [1] # initial neurons
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
xcs.PRED_EVOLVE_WEIGHTS = False
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

MIN_EXPLORE = 0.2 # the minimum exploration rate
EXPLORE_DECAY = 0.995 # the decay of exploration after each batch replay
N = 100 # maximum 10,000 learning trials/episodes
trials = np.zeros(N)
error = np.zeros(N)
memory = deque(maxlen = 200 * N)
scores = deque(maxlen = N)

frames = [] # for creating a gif
fscore = []
ftrial = []

def replay(replay_size = 200):
    """Performs experience replay updates"""
    batch = random.sample(memory, min(len(memory), replay_size))
    for state, action, reward, next_state, is_reset in batch:
        y_target = reward
        if not is_reset: # predict next state payoff
            prediction_array = xcs.predict(next_state.reshape(1, X_DIM))[0]
            y_target += xcs.GAMMA * np.max(prediction_array)
        xcs.init_trial()
        xcs.init_step()
        xcs.decision(state, True) # create match set and forward prop state
        xcs.update(y_target, True, action) # create action set and update
        xcs.end_step()
        xcs.end_trial()
    if xcs.P_EXPLORE > MIN_EXPLORE: # increasingly exploit
        xcs.P_EXPLORE *= EXPLORE_DECAY

# learning episodes
for i in range(N):
    for j in range(xcs.PERF_TRIALS):
        # explore trial/episode
        state = env.reset()
        xcs.init_trial()
        while True:
            xcs.init_step()
            action = xcs.decision(state, True)
            next_state, reward, is_reset, info = env.step(action)
            xcs.end_step()
            memory.append((state, action, reward, next_state, is_reset))
            state = next_state
            if is_reset:
                break
        xcs.end_trial()
        replay() # perform experience replay update
        # exploit trial/episode
        episode_score = 0
        cnt = 0
        err = 0
        state = env.reset()
        xcs.init_trial()
        while True:
            if SAVE_GIF and i % 5 == 0 and j == 0: # every 500 trials
                frames.append(env.render(mode="rgb_array"))
                fscore.append(episode_score)
                ftrial.append(i * xcs.PERF_TRIALS)
            xcs.init_step()
            action = xcs.decision(state, False)
            next_state, reward, is_reset, info = env.step(action)
            err += xcs.error(reward, is_reset, MAX_PAYOFF)
            episode_score += reward
            cnt += 1
            xcs.end_step()
            state = next_state
            if is_reset:
                if SAVE_GIF and i % 5 == 0 and j == 0:
                    for delay in range(100):
                        frames.append(frames[-1])
                        fscore.append(fscore[-1])
                        ftrial.append(ftrial[-1])
                break
        xcs.end_trial()
        scores.append(episode_score)
        error[i] += err / float(cnt)
    error[i] /= float(xcs.PERF_TRIALS)
    trials[i] = (i + 1) * xcs.PERF_TRIALS
    mean_score = np.mean(scores)
    status = ("episodes=%d score=%.2f error=%.5f" %
              (trials[i], mean_score, error[i]))
    print(status)
    if mean_score > env.spec.reward_threshold: # solved
        print("solved: score %.2f > %.2f" %
              (mean_score, env.spec.reward_threshold))
        break

# final exploit trial/episode
episode_score = 0
err = 0
cnt = 0
state = env.reset()
xcs.init_trial()
while True:
    if SAVE_GIF:
        frames.append(env.render(mode="rgb_array"))
        fscore.append(episode_score)
        ftrial.append('final')
    else:
        env.render()
    xcs.init_step()
    action = xcs.decision(state, False)
    next_state, reward, is_reset, info = env.step(action)
    err += xcs.error(reward, is_reset, MAX_PAYOFF)
    cnt += 1
    episode_score += reward
    xcs.end_step()
    state = next_state
    if is_reset:
        if SAVE_GIF:
            for delay in range(100):
                frames.append(frames[-1])
                fscore.append(fscore[-1])
                ftrial.append(ftrial[-1])
        break
xcs.end_trial()
perf = (episode_score / env._max_episode_steps) * 100
print("exploit: perf=%.2f%%, score=%.2f, error=%.5f" %
      (perf, episode_score, err / cnt))

# close Gym
env.close()

if SAVE_GIF:
    print("Creating gif. This may take a while...")
    save_frames_as_gif(frames, fscore)

# to crop and optimise gif
# gifsicle -O3 --colors=64 --use-col=web --lossy=100 --crop 10,10-270,220 --output out.gif animation.gif
