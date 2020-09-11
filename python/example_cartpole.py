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
This example demonstrates the use of experience replay with XCSF to solve the
cart-pole problem from the OpenAI Gym.
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

SAVE_GIF = False # for creating a gif
SAVE_GIF_EPISODES = 50
frames = []
fscore = []
ftrial = []

def save_frames_as_gif(path='./', filename='animation.gif'):
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
        text = ('episode = %3s, score = %3s' % (strial, sscore))
        ax.annotate(text, xy=(0,100), xytext=(-40,1), fontsize=12, bbox=bbox)
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
xcs.EPS_0 = 0.001 # target error
xcs.GAMMA = 0.95 # discount rate for delayed reward
xcs.BETA = 0.05 # classifier parameter update rate
xcs.ALPHA = 1 # accuracy offset
xcs.NU = 5 # accuracy slope
xcs.EA_SUBSUMPTION = False
xcs.SET_SUBSUMPTION = False
xcs.THETA_EA = 100 # EA invocation frequency
xcs.THETA_DEL = 100 # min experience before fitness used for deletion

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
xcs.PRED_NUM_NEURONS = [5] # initial neurons
xcs.PRED_MAX_NEURONS = [100] # maximum neurons
xcs.PRED_EVOLVE_WEIGHTS = True
xcs.PRED_EVOLVE_NEURONS = True
xcs.PRED_EVOLVE_FUNCTIONS = False
xcs.PRED_EVOLVE_CONNECTIVITY = False
xcs.PRED_EVOLVE_ETA = True
xcs.PRED_SGD_WEIGHTS = True
xcs.PRED_ETA = 0.0001 # maximum gradient descent rate
xcs.PRED_DECAY = 0 # weight decay

xcs.P_EXPLORE = 1 # initial probability of exploring on a learning step
MIN_EXPLORE = 0.01 # the minimum exploration rate
EXPLORE_DECAY = 0.995 # the decay of exploration after each batch replay

xcs.print_params()

#####################
# Execute experiment
#####################

MAX_EPISODES = 2000 # maximum number of episodes to run
N = 100 # number of episodes to average performance
memory = deque(maxlen = 50000)
scores = deque(maxlen = N)

def replay(replay_size = 5000):
    """Performs experience replay updates"""
    batch_size = min(len(memory), replay_size)
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        y_target = reward
        if not done:
            prediction_array = xcs.predict(next_state.reshape(1, X_DIM))[0]
            y_target += xcs.GAMMA * np.max(prediction_array)
        xcs.init_trial()
        xcs.init_step()
        xcs.decision(state, True) # create match set and forward prop state
        xcs.update(y_target, True, action) # create action set and update
        xcs.end_step()
        xcs.end_trial()
    if xcs.P_EXPLORE > MIN_EXPLORE:
        xcs.P_EXPLORE *= EXPLORE_DECAY

def episode(learn, episode_nr, gif):
    """Performs a single episode, recording learning episodes"""
    episode_score = 0
    steps = 0
    state = env.reset()
    xcs.init_trial()
    while True:
        xcs.init_step()
        action = xcs.decision(state, learn)
        next_state, reward, done, _ = env.step(action)
        xcs.end_step()
        steps += 1
        episode_score += reward
        if learn:
            memory.append((state, action, reward, next_state, done))
        if gif:
            frames.append(env.render(mode="rgb_array"))
            fscore.append(episode_score)
            ftrial.append(episode_nr)
        if done:
            if gif:
                for _ in range(100):
                    frames.append(frames[-1])
                    fscore.append(fscore[-1])
                    ftrial.append(ftrial[-1])
            break
        state = next_state
    xcs.end_trial()
    if not learn:
        scores.append(episode_score)
    return episode_score

# learning episodes
for j in range(MAX_EPISODES):
    # learning episode
    episode(True, j, False)
    # experience replay update
    replay()
    # exploit episode for monitoring performance
    if SAVE_GIF and j % SAVE_GIF_EPISODES == 0:
        episode(False, j, True)
    else:
        episode(False, j, False)
    mean_score = np.mean(scores)
    print ("episodes=%d mean_score=%.2f" % (j, mean_score))
    # is the problem solved?
    if j > 99 and mean_score > env.spec.reward_threshold:
        print("solved after %d episodes: mean score %.2f > %.2f" %
              (j, mean_score, env.spec.reward_threshold))
        break

# final exploit episode
score = episode(False, j, SAVE_GIF)
perf = (score / env._max_episode_steps) * 100
print("exploit: perf=%.2f%%, score=%.2f" % (perf, score))

# close Gym
env.close()

if SAVE_GIF:
    print("Creating gif. This may take a while...")
    save_frames_as_gif()
    print("To crop and optimise gif:")
    print("gifsicle -O3 --colors=64 --use-col=web --lossy=100 " \
          "--crop 0,10-270,220 --output out.gif animation.gif")
