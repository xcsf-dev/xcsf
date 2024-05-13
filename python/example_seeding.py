#!/usr/bin/python3
#
# Copyright (C) 2021--2023 Richard Preen <rpreen@gmail.com>
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
This example demonstrates the insertion of a human-defined classifier
into the XCSF population set.
"""

import json

import numpy as np

import xcsf

RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)

##############################################################################
# Human-designed classifier.
##############################################################################

classifier = {
    "error": 10,  # each of these properties are optional
    "fitness": 1.01,
    "accuracy": 2,
    "set_size": 100,
    "numerosity": 2,
    "experience": 3,
    "time": 3,
    "samples_seen": 2,
    "samples_matched": 1,
    "condition": {
        "type": "hyperrectangle_csr",
        "center": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "spread": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "mutation": [0.2],  # this parameter still self-adapts
    },
    "action": {"type": "integer", "action": 0, "mutation": [0.28]},
    # prediction is absent and therefore initialised as normal
}

##############################################################################
# Write a population set file to demonstrate initialisation from a file.
# Note: an existing population can be saved with xcs.json_write("pset.json")
##############################################################################

with open("pset.json", "w", encoding="utf-8") as file:
    pset = {"classifiers": [classifier]}
    json.dump(pset, file)

##############################################################################
# Initialise XCSF.
##############################################################################

xcs = xcsf.XCS(
    x_dim=8,
    y_dim=1,
    n_actions=1,
    random_state=RANDOM_STATE,
    population_file="pset.json",  # optional population set file
    max_trials=1,
    pop_init=False,  # don't add any extra random classifiers
    action={"type": "integer"},
    condition={"type": "hyperrectangle_csr"},
    prediction={"type": "nlms_linear"},
)

##############################################################################
# Add the human-designed classifier again, this time manually.
##############################################################################

json_str = json.dumps(classifier)  # dictionary to JSON

# The json_insert_cl() function can be used to insert a single new classifier
# into the population. The new classifier is initialised with a random
# condition, action, prediction, and then any supplied properties overwrite
# these values. This means that all properties are optional. If the population
# set numerosity exceeds xcs.POP_SIZE after inserting the rule, the standard
# roulette wheel deletion mechanism will be invoked to maintain the population
# limit.
xcs.json_insert_cl(json_str)  # insert in [P]

##############################################################################

print("******************************")
print("BEFORE FITTING")
print("******************************")

xcs.print_pset()

##############################################################################
# Fit a single sample.
##############################################################################

X = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).reshape(1, -1)
y = np.random.random(1)

xcs.fit(X, y, warm_start=True)  # use existing population

##############################################################################

print("******************************")
print("AFTER FITTING")
print("******************************")

xcs.print_pset()
