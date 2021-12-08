#
# Copyright (C) 2020 Richard Preen <rpreen@gmail.com> Copyright (C) 2021 David
# Pätzel
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
#

set(XCSF_TESTS
    cond_ellipsoid_test.cpp
    cond_rectangle_test.cpp
    cond_ternary_test.cpp
    loss_test.cpp
    neural_layer_connected_test.cpp
    neural_layer_convolutional_test.cpp
    neural_layer_lstm_test.cpp
    neural_layer_maxpool_test.cpp
    neural_layer_recurrent_test.cpp
    neural_test.cpp
    pred_nlms_test.cpp
    pred_rls_test.cpp
    util_test.cpp
    unit_tests.cpp)

add_definitions(-DDSFMT_MEXP=19937)

add_executable(tests ${XCSF_TESTS})
target_link_libraries(tests xcs)

add_test(NAME xcsf COMMAND tests)

add_custom_command(
  TARGET tests
  COMMENT "Running tests..."
  POST_BUILD
  COMMAND tests)
