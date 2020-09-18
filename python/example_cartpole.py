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
    """ Save animation as gif """
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

xcs.OMP_NUM_THREADS = 8 # number of CPU cores to use
xcs.POP_SIZE = 500 # maximum population size
xcs.EPS_0 = 0.001 # target error
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

GAMMA = 0.95 # discount rate for delayed reward
epsilon = 1 # initial probability of exploring
EPSILON_MIN = 0.1 # the minimum exploration rate
EPSILON_DECAY = 0.9 # the decay of exploration after each batch replay

xcs.print_params()

#####################
# Execute experiment
#####################

total_steps = 0 # total number of steps performed
MAX_EPISODES = 2000 # maximum number of episodes to run
N = 100 # number of episodes to average performance
memory = deque(maxlen = 50000) # memory buffer for experience replay
scores = deque(maxlen = N) # scores used to calculate moving average

def replay(replay_size=5000):
    """ Performs experience replay updates """
    batch_size = min(len(memory), replay_size)
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        y_target = reward
        if not done:
            prediction_array = xcs.predict(next_state.reshape(1,-1))[0]
            y_target += GAMMA * np.max(prediction_array)
        xcs.update_sar(state, action, y_target)

def egreedy_action(state):
    """ Selects an action using an epsilon greedy policy """
    if np.random.rand() < epsilon:
        return random.randrange(N_ACTIONS)
    prediction_array = xcs.predict(state.reshape(1,-1))[0]
    return np.argmax(prediction_array)

def episode(episode_nr, create_gif):
    """ Executes a single episode, saving to memory buffer """
    episode_score = 0
    episode_steps = 0
    state = env.reset()
    while True:
        action = egreedy_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        episode_score += reward
        memory.append((state, action, reward, next_state, done))
        if create_gif:
            frames.append(env.render(mode="rgb_array"))
            fscore.append(episode_score)
            ftrial.append(episode_nr)
        if done:
            if create_gif:
                for _ in range(100):
                    frames.append(frames[-1])
                    fscore.append(fscore[-1])
                    ftrial.append(ftrial[-1])
            break
        state = next_state
    return episode_score, episode_steps

# learning episodes
for ep in range(MAX_EPISODES):
    gif = False
    if SAVE_GIF and ep % SAVE_GIF_EPISODES == 0:
        gif = True
    # execute a single episode
    ep_score, ep_steps = episode(ep, gif)
    # perform experience replay updates
    replay()
    # display performance
    total_steps += ep_steps
    scores.append(ep_score)
    mean_score = np.mean(scores)
    print ("episodes=%d steps=%d score=%.2f epsilon=%.5f" %
           (ep, total_steps, mean_score, epsilon))
    # is the problem solved?
    if ep > 99 and mean_score > env.spec.reward_threshold:
        print("solved after %d episodes: mean score %.2f > %.2f" %
              (ep, mean_score, env.spec.reward_threshold))
        break
    # decay the exploration rate
    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

# final exploit episode
epsilon = 0
ep_score, ep_steps = episode(ep, SAVE_GIF)
perf = (ep_score / env._max_episode_steps) * 100
print("exploit: perf=%.2f%%, score=%.2f" % (perf, ep_score))

# close Gym
env.close()

if SAVE_GIF:
    print("Creating gif. This may take a while...")
    save_frames_as_gif()
    print("To crop and optimise gif:")
    print("gifsicle -O3 --colors=64 --use-col=web --lossy=100 " \
          "--crop 0,10-270,220 --output out.gif animation.gif")
