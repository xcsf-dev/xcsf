#!/usr/bin/python3
#
# Copyright (C) 2021--2022 Richard Preen <rpreen@gmail.com>
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

from __future__ import annotations

import json
from typing import Final

import xcsf

xcs = xcsf.XCS(8, 1, 2)  # (x_dim, y_dim, n_actions)
xcs.condition("hyperrectangle_csr")
xcs.action("integer")
xcs.prediction("nlms_linear")

classifier: Final[dict] = {
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
    "action": {"type": "integer", "action": 1, "mutation": [0.28]}
    # prediction is absent and therefore initialised as normal
}

json_str = json.dumps(classifier)  # dictionary to JSON

# The json_insert_cl() function can be used to insert a single new classifier
# into the population. The new classifier is initialised with a random
# condition, action, prediction, and then any supplied properties overwrite
# these values. This means that all properties are optional. If the population
# set numerosity exceeds xcs.POP_SIZE after inserting the rule, the standard
# roulette wheel deletion mechanism will be invoked to maintain the population
# limit.
xcs.json_insert_cl(json_str)  # insert in [P]

xcs.print_pset(True, True, True)
