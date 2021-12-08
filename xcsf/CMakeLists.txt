#
# Copyright (C) 2019--2021 Richard Preen <rpreen@gmail.com>
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

set(XCSF_SOURCES
    act_integer.c
    act_neural.c
    action.c
    blas.c
    cl.c
    clset.c
    clset_neural.c
    cond_dgp.c
    cond_dummy.c
    cond_ellipsoid.c
    cond_gp.c
    cond_neural.c
    cond_rectangle.c
    cond_ternary.c
    condition.c
    config.c
    dgp.c
    ea.c
    env.c
    env_csv.c
    env_maze.c
    env_mux.c
    gp.c
    image.c
    loss.c
    neural.c
    neural_activations.c
    neural_layer.c
    neural_layer_args.c
    neural_layer_avgpool.c
    neural_layer_connected.c
    neural_layer_convolutional.c
    neural_layer_dropout.c
    neural_layer_lstm.c
    neural_layer_maxpool.c
    neural_layer_noise.c
    neural_layer_recurrent.c
    neural_layer_softmax.c
    neural_layer_upsample.c
    pa.c
    param.c
    perf.c
    pred_constant.c
    pred_neural.c
    pred_nlms.c
    pred_rls.c
    prediction.c
    rule_dgp.c
    rule_neural.c
    sam.c
    utils.c
    xcs_rl.c
    xcs_supervised.c
    xcsf.c)

set(XCSF_HEADERS
    act_integer.h
    act_neural.h
    action.h
    blas.h
    cl.h
    clset.h
    clset_neural.h
    cond_dgp.h
    cond_dummy.h
    cond_ellipsoid.h
    cond_gp.h
    cond_neural.h
    cond_rectangle.h
    cond_ternary.h
    condition.h
    config.h
    dgp.h
    ea.h
    env.h
    env_csv.h
    env_maze.h
    env_mux.h
    gp.h
    image.h
    loss.h
    neural.h
    neural_activations.h
    neural_layer.h
    neural_layer_args.h
    neural_layer_avgpool.h
    neural_layer_connected.h
    neural_layer_convolutional.h
    neural_layer_dropout.h
    neural_layer_lstm.h
    neural_layer_maxpool.h
    neural_layer_noise.h
    neural_layer_recurrent.h
    neural_layer_softmax.h
    neural_layer_upsample.h
    pa.h
    param.h
    perf.h
    pred_constant.h
    pred_neural.h
    pred_nlms.h
    pred_rls.h
    prediction.h
    rule_dgp.h
    rule_neural.h
    sam.h
    utils.h
    xcs_rl.h
    xcs_supervised.h
    xcsf.h)

set(DSFMT
    ${CMAKE_SOURCE_DIR}/lib/dSFMT/dSFMT.c
    ${CMAKE_SOURCE_DIR}/lib/dSFMT/dSFMT.h
    ${CMAKE_SOURCE_DIR}/lib/dSFMT/dSFMT-common.h
    ${CMAKE_SOURCE_DIR}/lib/dSFMT/dSFMT-params.h
    ${CMAKE_SOURCE_DIR}/lib/dSFMT/dSFMT-params19937.h)

set(CJSON ${CMAKE_SOURCE_DIR}/lib/cJSON/cJSON.c
          ${CMAKE_SOURCE_DIR}/lib/cJSON/cJSON.h)

# ##############################################################################
# target: libxcs - main functions
# ##############################################################################

add_definitions(-DDSFMT_MEXP=19937)
add_library(xcs STATIC ${XCSF_SOURCES} ${XCSF_HEADERS} ${DSFMT} ${CJSON})
target_link_libraries(xcs PUBLIC m)
if(PARALLEL AND OpenMP_FOUND)
  target_link_libraries(xcs PUBLIC OpenMP::OpenMP_C)
endif()

# ##############################################################################
# target: main - stand-alone binary execution
# ##############################################################################

if(XCSF_MAIN)
  add_executable(main main.c)
  target_link_libraries(main PUBLIC xcs)
endif()

# ##############################################################################
# target: xcsf.so / pyd - Python library
# ##############################################################################

if(XCSF_PYLIB)
  set(XCSF_PY_SOURCES pybind_wrapper.cpp)
  pybind11_add_module(xcsf ${XCSF_PY_SOURCES})
  if(PARALLEL AND OpenMP_FOUND)
    target_link_libraries(xcsf PUBLIC OpenMP::OpenMP_CXX)
  endif()
  if(WIN32) # https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html
    target_link_libraries(
      xcsf
      PUBLIC xcs
             -static-libgcc
             -static-libstdc++
             -Wl,-Bstatic,--whole-archive
             -lwinpthread
             -Wl,--no-whole-archive
             -lgomp
             -Wl,--no-whole-archive)
  else()
    target_link_libraries(xcsf PUBLIC xcs)
  endif()
endif()
