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

import gymnasium as gym
import numpy as np

import xcsf

RANDOM_STATE: int = 1010
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

############################################
# Initialise OpenAI Gym problem environment
############################################

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset(seed=RANDOM_STATE)

X_DIM: int = int(env.observation_space.shape[0])
N_ACTIONS: int = int(env.action_space.n)

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
    pop_size=200,
    theta_del=100,
    e0=0.001,
    alpha=1,
    beta=0.05,
    ea={
        "select_type": "roulette",
        "theta_ea": 100,
        "lambda": 2,
    },
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

GAMMA: float = 0.95  # discount rate for delayed reward
epsilon: float = 1  # initial probability of exploring
EPSILON_MIN: float = 0.1  # the minimum exploration rate
EPSILON_DECAY: float = 0.98  # the decay of exploration after each batch replay
REPLAY_TIME: int = 1  # perform replay update every n episodes

print(json.dumps(xcs.internal_params(), indent=4))

#####################
# Execute experiment
#####################

total_steps: int = 0  # total number of steps performed
MAX_EPISODES: int = 2000  # maximum number of episodes to run
N: int = 100  # number of episodes to average performance
memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=50000)
scores: deque[float] = deque(maxlen=N)  # used to calculate moving average


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


def episode() -> tuple[float, int]:
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
        if done:
            break
        state = next_state
    return episode_score, episode_steps


# learning episodes
for ep in range(MAX_EPISODES):
    # execute a single episode
    ep_score, ep_steps = episode()
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
ep_score, ep_steps = episode()

# close Gym
env.close()
