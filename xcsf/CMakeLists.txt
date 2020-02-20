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

set(SOURCE
    act_integer.c
    act_integer.h
    action.c
    action.h
    cl.c
    cl.h
    clset.c
    clset.h
    cond_dgp.c
    cond_dgp.h
    cond_dummy.c
    cond_dummy.h
    cond_ellipsoid.c
    cond_ellipsoid.h
    cond_gp.c
    cond_gp.h
    cond_neural.c
    cond_neural.h
    cond_rectangle.c
    cond_rectangle.h
    cond_ternary.c
    cond_ternary.h
    condition.c
    condition.h
    config.c
    config.h
    dgp.c
    dgp.h
    ea.c
    ea.h
    env.c
    env.h
    env_csv.c
    env_csv.h
    env_maze.c
    env_maze.h
    env_mux.c
    env_mux.h
    blas.c
    blas.h
    gp.c
    gp.h
    loss.c
    loss.h
    main.c
    mt19937-64.c
    mt64.h
    neural.c
    neural.h
    neural_activations.c
    neural_activations.h
    neural_layer.c
    neural_layer.h
    neural_layer_connected.c
    neural_layer_connected.h
    neural_layer_dropout.c
    neural_layer_dropout.h
    neural_layer_noise.c
    neural_layer_noise.h
    neural_layer_softmax.c
    neural_layer_softmax.h
    pa.c
    pa.h
    perf.c
    perf.h
    pred_constant.c
    pred_constant.h
    pred_neural.c
    pred_neural.h
    pred_nlms.c
    pred_nlms.h
    pred_rls.c
    pred_rls.h
    prediction.c
    prediction.h
    rule_dgp.c
    rule_dgp.h
    rule_neural.c
    rule_neural.h
    sam.c
    sam.h
    utils.c
    utils.h
    xcs_multi_step.c
    xcs_multi_step.h
    xcs_single_step.c
    xcs_single_step.h
    xcsf.c
    xcsf.h
)

##########################
# GPU files
##########################

if(GPU)

    set(SOURCE
        ${SOURCE}
        cuda.cu
        cuda.h
        blas_kernels.cu
        blas_kernels.h
        neural_activation_kernels.cu
        neural_activation_kernels.h
    )

endif()

##########################
# Stand-alone binary
##########################

if(NOT XCSF_PYLIB)

    add_executable(xcsf ${SOURCE})
    target_link_libraries(xcsf m)

##########################
# Python library
##########################

else()

    set(SOURCE
        ${SOURCE}
        python_wrapper.cpp
    )

    PYTHON_ADD_MODULE(xcsf ${SOURCE})
    file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)

endif()
