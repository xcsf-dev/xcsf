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
    bbox = dict(boxstyle='round', fc='0.8')
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

# constructor = (x_dim, y_dim, n_actions)
xcs = xcsf.XCS(X_DIM, N_ACTIONS, 1) # Supervised mode: i.e, single action

xcs.OMP_NUM_THREADS = 8 # number of CPU cores to use
xcs.POP_INIT = False # use covering to initialise
xcs.MAX_TRIALS = 1 # one trial per fit
xcs.POP_SIZE = 200 # maximum population size
xcs.EPS_0 = 0.001 # target error
xcs.BETA = 0.05 # classifier parameter update rate
xcs.ALPHA = 1 # accuracy offset
xcs.NU = 5 # accuracy slope
xcs.EA_SUBSUMPTION = False
xcs.SET_SUBSUMPTION = False
xcs.THETA_EA = 100 # EA invocation frequency
xcs.THETA_DEL = 100 # min experience before fitness used for deletion

condition_layers = {
    'layer_0': { # hidden layer
        'type': 'connected',
        'activation': 'selu',
        'evolve-weights': True,
        'evolve-neurons': True,
        'n-init': 1,
        'n-max': 100,
        'max-neuron-grow': 1,
    },
    'layer_1': { # output layer
        'type': 'connected',
        'activation': 'linear',
        'evolve-weights': True,
        'n-init': 1,
    }
}

xcs.condition('neural', condition_layers) # neural network conditions
xcs.action('integer') # (dummy) integer actions
xcs.prediction('rls-quadratic') # Quadratic RLS

GAMMA = 0.95 # discount rate for delayed reward
epsilon = 1 # initial probability of exploring
EPSILON_MIN = 0.1 # the minimum exploration rate
EPSILON_DECAY = 0.98 # the decay of exploration after each batch replay
REPLAY_TIME = 1 # perform replay update every n episodes

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
        target = xcs.predict(state.reshape(1,-1))[0]
        target[action] = y_target
        xcs.fit(state.reshape(1,-1), target.reshape(1,-1), True)

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
            frames.append(env.render(mode='rgb_array'))
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
    if ep % REPLAY_TIME == 0:
        replay()
    # display performance
    total_steps += ep_steps
    scores.append(ep_score)
    mean_score = np.mean(scores)
    print('episodes=%d steps=%d score=%.2f epsilon=%.5f error=%.5f msize=%.2f' %
          (ep, total_steps, mean_score, epsilon, xcs.error(), xcs.msetsize()))
    # is the problem solved?
    if ep > N and mean_score > env.spec.reward_threshold:
        print('solved after %d episodes: mean score %.2f > %.2f' %
              (ep, mean_score, env.spec.reward_threshold))
        break
    # decay the exploration rate
    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

# final exploit episode
epsilon = 0
ep_score, ep_steps = episode(ep, SAVE_GIF)

# close Gym
env.close()

if SAVE_GIF:
    print('Creating gif. This may take a while...')
    save_frames_as_gif()
    print('To crop and optimise gif:')
    print('gifsicle -O3 --colors=64 --use-col=web --lossy=100 ' \
          '--crop 0,10-270,220 --output out.gif animation.gif')
