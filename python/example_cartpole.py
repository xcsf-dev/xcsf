#!/usr/bin/python3
#
# Copyright (C) 2020--2021 Richard Preen <rpreen@gmail.com>
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

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rcParams

import xcsf

############################################
# Initialise OpenAI Gym problem environment
############################################

env = gym.make("CartPole-v0")
X_DIM: int = env.observation_space.shape[0]
N_ACTIONS: int = env.action_space.n

SAVE_GIF: bool = False  # for creating a gif
SAVE_GIF_EPISODES: int = 50
frames: List[List[float]] = []
fscore: List[float] = []
ftrial: List[int] = []


def save_frames_as_gif(path: str = "./", filename: str = "animation.gif") -> None:
    """Save animation as gif"""
    rcParams["font.family"] = "monospace"
    fig = plt.figure(dpi=90)
    fig.set_size_inches(3, 3)
    ax = fig.add_subplot(111)
    patch = plt.imshow(frames[0])
    bbox = dict(boxstyle="round", fc="0.8")
    plt.axis("off")

    def animate(i: int) -> None:
        patch.set_data(frames[i])
        strial = str(ftrial[i])
        sscore = str(int(fscore[i]))
        text = "episode = %3s, score = %3s" % (strial, sscore)
        ax.annotate(text, xy=(0, 100), xytext=(-40, 1), fontsize=12, bbox=bbox)

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=100, blit=False
    )
    anim.save(path + filename, writer="imagemagick", fps=30)


###################
# Initialise XCSF
###################

# constructor = (x_dim, y_dim, n_actions)
xcs: xcsf.XCS = xcsf.XCS(X_DIM, N_ACTIONS, 1)  # Supervised: i.e, single action

xcs.OMP_NUM_THREADS = 8  # number of CPU cores to use
xcs.POP_INIT = False  # use covering to initialise
xcs.MAX_TRIALS = 1  # one trial per fit
xcs.POP_SIZE = 200  # maximum population size
xcs.E0 = 0.001  # target error
xcs.BETA = 0.05  # classifier parameter update rate
xcs.ALPHA = 1  # accuracy offset
xcs.NU = 5  # accuracy slope
xcs.EA_SUBSUMPTION = False
xcs.SET_SUBSUMPTION = False
xcs.THETA_EA = 100  # EA invocation frequency
xcs.THETA_DEL = 100  # min experience before fitness used for deletion

condition_layers: dict = {
    "layer_0": {  # hidden layer
        "type": "connected",
        "activation": "selu",
        "evolve_weights": True,
        "evolve_neurons": True,
        "n_init": 1,
        "n_max": 100,
        "max_neuron_grow": 1,
    },
    "layer_1": {  # output layer
        "type": "connected",
        "activation": "linear",
        "evolve_weights": True,
        "n_init": 1,
    },
}

xcs.condition("neural", condition_layers)  # neural network conditions
xcs.action("integer")  # (dummy) integer actions
xcs.prediction("rls_quadratic")  # Quadratic RLS

GAMMA: float = 0.95  # discount rate for delayed reward
epsilon: float = 1  # initial probability of exploring
EPSILON_MIN: float = 0.1  # the minimum exploration rate
EPSILON_DECAY: float = 0.98  # the decay of exploration after each batch replay
REPLAY_TIME: int = 1  # perform replay update every n episodes

xcs.print_params()

#####################
# Execute experiment
#####################

total_steps: int = 0  # total number of steps performed
MAX_EPISODES: int = 2000  # maximum number of episodes to run
N: int = 100  # number of episodes to average performance
memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=50000)
scores: Deque[float] = deque(maxlen=N)  # used to calculate moving average


def replay(replay_size: int = 5000) -> None:
    """Performs experience replay updates"""
    batch_size: int = min(len(memory), replay_size)
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        y_target = reward
        if not done:
            prediction_array = xcs.predict(next_state.reshape(1, -1))[0]
            y_target += GAMMA * np.max(prediction_array)
        target = xcs.predict(state.reshape(1, -1))[0]
        target[action] = y_target
        xcs.fit(state.reshape(1, -1), target.reshape(1, -1), True)


def egreedy_action(state: np.ndarray) -> int:
    """Selects an action using an epsilon greedy policy"""
    if np.random.rand() < epsilon:
        return random.randrange(N_ACTIONS)
    prediction_array = xcs.predict(state.reshape(1, -1))[0]
    return int(np.argmax(prediction_array))


def episode(episode_nr: int, create_gif: bool) -> Tuple[float, int]:
    """Executes a single episode, saving to memory buffer"""
    episode_score: int = 0
    episode_steps: int = 0
    state: np.ndarray = env.reset()
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
    if ep % REPLAY_TIME == 0:
        replay()
    # display performance
    total_steps += ep_steps
    scores.append(ep_score)
    mean_score = np.mean(scores)
    print(
        "episodes=%d steps=%d score=%.2f epsilon=%.5f error=%.5f msize=%.2f"
        % (ep, total_steps, mean_score, epsilon, xcs.error(), xcs.mset_size())
    )
    # is the problem solved?
    if ep > N and mean_score > env.spec.reward_threshold:
        print(
            "solved after %d episodes: mean score %.2f > %.2f"
            % (ep, mean_score, env.spec.reward_threshold)
        )
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
    print("Creating gif. This may take a while...")
    save_frames_as_gif()
    print("To crop and optimise gif:")
    print(
        "gifsicle -O3 --colors=64 --use-col=web --lossy=100 "
        "--crop 0,10-270,220 --output out.gif animation.gif"
    )
