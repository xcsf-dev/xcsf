#!/usr/bin/python3
#
# Copyright (C) 2020--2023 Richard Preen <rpreen@gmail.com>
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
cart-pole problem from the OpenAI gymnasium.

$ pip install gymnasium[classic-control]

Note: These hyperparameters do not result in consistently optimal performance.
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Final

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from tqdm import tqdm

import xcsf

RANDOM_STATE: Final[int] = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

############################################
# Initialise OpenAI Gym problem environment
############################################

env = gym.make("CartPole-v1", render_mode="rgb_array")
X_DIM: Final[int] = int(env.observation_space.shape[0])
N_ACTIONS: Final[int] = int(env.action_space.n)

SAVE_GIF: Final[bool] = False  # for creating a gif
SAVE_GIF_EPISODES: Final[int] = 50
frames: list[list[float]] = []
fscore: list[float] = []
ftrial: list[int] = []

###################
# Initialise XCSF
###################

xcs = xcsf.XCS(
    x_dim=X_DIM,
    y_dim=N_ACTIONS,
    n_actions=1,
    omp_num_threads=12,
    random_state=RANDOM_STATE,
    pop_init=False,
    max_trials=1,  # one trial per fit()
    perf_trials=1,
    pop_size=200,
    loss_func="mse",
    e0=0.001,
    alpha=1,
    beta=0.05,
    condition={
        "type": "neural",
        "args": {
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
        },
    },
    prediction={
        "type": "rls_quadratic",
    },
)

GAMMA: Final[float] = 0.95  # discount rate for delayed reward
epsilon: float = 1  # initial probability of exploring
EPSILON_MIN: Final[float] = 0.1  # the minimum exploration rate
EPSILON_DECAY: Final[float] = 0.98  # the decay of exploration after each batch replay
REPLAY_TIME: Final[int] = 1  # perform replay update every n episodes

print(json.dumps(xcs.get_params(), indent=4))

#####################
# Execute experiment
#####################

total_steps: int = 0  # total number of steps performed
MAX_EPISODES: Final[int] = 2000  # maximum number of episodes to run
N: Final[int] = 100  # number of episodes to average performance
memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=50000)
scores: deque[float] = deque(maxlen=N)  # used to calculate moving average


def replay(replay_size: int = 5000) -> None:
    """Performs experience replay updates"""
    batch_size: Final[int] = min(len(memory), replay_size)
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        y_target = reward
        if not done:
            prediction_array = xcs.predict(next_state.reshape(1, -1))[0]
            y_target += GAMMA * np.max(prediction_array)
        target = xcs.predict(state.reshape(1, -1))[0]
        target[action] = y_target
        xcs.fit(
            state.reshape(1, -1), target.reshape(1, -1), warm_start=True, verbose=False
        )


def egreedy_action(state: np.ndarray) -> int:
    """Selects an action using an epsilon greedy policy"""
    if np.random.rand() < epsilon:
        return random.randrange(N_ACTIONS)
    prediction_array = xcs.predict(state.reshape(1, -1))[0]
    # break ties randomly
    best_actions = np.where(prediction_array == prediction_array.max())[0]
    return int(np.random.choice(best_actions))


def episode(episode_nr: int, create_gif: bool) -> tuple[float, int]:
    """Executes a single episode, saving to memory buffer"""
    episode_score: float = 0
    episode_steps: int = 0
    state: np.ndarray = env.reset()[0]
    while True:
        action = egreedy_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
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
    gif: bool = False
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
        f"episodes={ep} "
        f"steps={total_steps} "
        f"score={mean_score:.2f} "
        f"epsilon={epsilon:.5f} "
        f"error={xcs.error():.5f} "
        f"msize={xcs.mset_size():.2f}"
    )
    # is the problem solved?
    if ep > N and mean_score > env.spec.reward_threshold:
        print(
            f"solved after {ep} episodes: "
            f"mean score {mean_score:.2f} > {env.spec.reward_threshold:.2f}"
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
    # add score and episode nr
    rcParams["font.family"] = "monospace"
    bbox = dict(boxstyle="round", fc="0.8")
    annotated_frames = list()
    bar = tqdm(total=len(frames), position=0, leave=True)
    for i in range(len(frames)):
        fig = plt.figure(dpi=90)
        fig.set_size_inches(3, 3)
        ax = fig.add_subplot(111)
        plt.imshow(frames[i])
        plt.axis("off")
        strial = str(ftrial[i])
        sscore = str(int(fscore[i]))
        text = f"episode = {strial:3s}, score = {sscore:3s}"
        ax.annotate(text, xy=(0, 100), xytext=(-40, 1), fontsize=12, bbox=bbox)
        fig.canvas.draw()
        annotated_frames.append(np.asarray(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)
        bar.refresh()
        bar.update(1)
    bar.close()
    # write gif
    imageio.mimsave("animation.gif", annotated_frames, duration=30)
