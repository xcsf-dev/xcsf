#
# Copyright (C) 2019--2024 Richard Preen <rpreen@gmail.com>
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

cmake_minimum_required(VERSION 3.12)

project(XCSF CXX C)
set(PROJECT_VENDOR "Richard Preen")
set(PROJECT_CONTACT "rpreen@gmail.com")
set(PROJECT_URL "https://github.com/xcsf-dev/xcsf")
set(PROJECT_DESCRIPTION "XCSF: Learning Classifier System")
set(PROJECT_VERSION "1.4.7")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

option(XCSF_MAIN "Build XCSF stand-alone main executable" ON)
option(XCSF_PYLIB "Build XCSF Python library" OFF)
option(PARALLEL "Parallel match set and prediction" ON)
option(ENABLE_TESTS "Build standard unit tests" OFF)
option(PYTEST "Build Python tests" OFF)
option(NATIVE_OPT "Optimise for the native architecture" ON)
option(ENABLE_DOXYGEN "Enable Building XCSF Documentation" ON)
option(GEN_PROF "Generate profiling information" OFF)
option(USE_PROF "Use profiling information" OFF)
option(USE_GCOV "Generate test coverage analysis" OFF)
option(SANITIZE "Build with sanitizers" OFF)

if(NOT MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -Wall -Wextra ")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused ")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfatal-errors")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-qual")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wredundant-decls")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winit-self")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-align")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wlogical-op")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-declarations")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-prototypes")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wnested-externs")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wold-style-definition")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpointer-arith")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-prototypes")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wundef")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wuninitialized")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmaybe-uninitialized")
  endif()
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(GEN_PROF)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fprofile-generate")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate")
endif()

if(USE_PROF)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-use")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-correction")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-correction")
endif()

if(USE_GCOV)
  find_program(GENHTML genhtml)
  find_program(LCOV lcov)
  if(NOT LCOV OR NOT GENHTML)
    message(SEND_ERROR "Coverage analysis requires lcov and genhtml.")
  endif()
  if(NOT ENABLE_TESTS)
    set(ENABLE_TESTS ON)
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
endif()

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(PARALLEL)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_MATCH")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_PRED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_UPDATE")
  endif()
endif()

if(UNIX
   AND NOT APPLE
   AND CMAKE_C_COMPILER_ID MATCHES "Clang")
  # Linux Clang
  set(CMAKE_AR
      ${CMAKE_CXX_COMPILER_AR}
      CACHE PATH "AR" FORCE)
  set(CMAKE_RANLIB
      ${CMAKE_CXX_COMPILER_RANLIB}
      CACHE PATH "RANLIB" FORCE)
endif()

set(CMAKE_C_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NATIVE_OPT)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")  # Not on Apple Silicon
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  endif()
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")

if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffat-lto-objects")
  set(CMAKE_C_FLAGS_RELEASE
      "${CMAKE_C_FLAGS_RELEASE} -fno-semantic-interposition")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffat-lto-objects")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
endif()

if(XCSF_MAIN)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cfg/default.json DESTINATION .)
endif()

if(XCSF_PYLIB)
  add_subdirectory(lib/pybind11)
  file(GLOB PYTHON_EXAMPLES "python/*.py")
  file(COPY ${PYTHON_EXAMPLES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(COPY "xcsf/utils/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/xcsf/utils/)
  file(COPY "xcsf/__init__.py" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/xcsf/)
  if(PYTEST)
    # Copy the Python tests
    file(GLOB PYTHON_TESTS "test/python/*.py")
    file(COPY ${PYTHON_TESTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    # Create a virtual environment
    set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv")
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -m venv "${VENV_DIR}")
    # Activate the virtual environment
    if(WIN32)
      set(PYTHON_EXEC "${VENV_DIR}/Scripts/python")
    else()
      set(PYTHON_EXEC "${VENV_DIR}/bin/python")
    endif()
    # Install dependencies
    execute_process(
      COMMAND "${PYTHON_EXEC}" -m pip install pytest numpy scikit-learn
    )
    # Add pytest to ctest (to capture CI failures)
    add_test(NAME pytest COMMAND "${PYTHON_EXEC}" -m pytest .)
    # Execute pytest at the end of building
    add_custom_target(run_tests ALL
      COMMAND "${PYTHON_EXEC}" -m pytest .
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Running pytest..."
    )
    # Wait for XCSF to be built
    add_dependencies(run_tests xcsf)
  endif()
endif()

if(ENABLE_DOXYGEN)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(
      doc # ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM)
  endif()
endif()

add_subdirectory(xcsf)

message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
