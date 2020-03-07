# 
#  Copyright (C) 2019--2020 Richard Preen <rpreen@gmail.com>
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

set(XCSF_SOURCES
    act_integer.c
    action.c
    blas.c
    cl.c
    clset.c
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
    loss.c
    neural.c
    neural_activations.c
    neural_layer.c
    neural_layer_connected.c
    neural_layer_dropout.c
    neural_layer_noise.c
    neural_layer_softmax.c
    pa.c
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
    xcs_multi_step.c
    xcs_single_step.c
    xcs_supervised.c
    xcsf.c
)

set(XCSF_HEADERS
    act_integer.h
    action.h
    blas.h
    cl.h
    clset.h
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
    loss.h
    neural.h
    neural_activations.h
    neural_layer.h
    neural_layer_connected.h
    neural_layer_dropout.h
    neural_layer_noise.h
    neural_layer_softmax.h
    pa.h
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
    xcs_multi_step.h
    xcs_single_step.h
    xcs_supervised.h
    xcsf.h
)

set(dSFMT_SOURCES
    ../lib/dSFMT/dSFMT.c
)

set(dSFMT_HEADERS
    ../lib/dSFMT/dSFMT-common.h
    ../lib/dSFMT/dSFMT-params.h
    ../lib/dSFMT/dSFMT-params19937.h
    ../lib/dSFMT/dSFMT.h
)
#################################################
# target: libxcs - main functions
#################################################

add_library(xcs STATIC ${XCSF_SOURCES} ${XCSF_HEADERS} ${dSFMT_SOURCES} ${dSFMT_HEADERS})
target_link_libraries(xcs m)

#################################################
# target: main - standalone binary execution
#################################################

add_executable(main main.c)
target_link_libraries(main xcs)

#################################################
# target: xcsf.so / pyd - Python library
#################################################

if(XCSF_PYLIB)

    set(XCSF_PY_SOURCES
        pybind_wrapper.cpp
    )

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ")
    PYTHON_ADD_MODULE(xcsf SHARED ${XCSF_PY_SOURCES})
    target_link_libraries(xcsf xcs)
    file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)

endif()
