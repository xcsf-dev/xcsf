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

cmake_minimum_required(VERSION 3.12)

project(XCSF CXX C)
set(PROJECT_VENDOR "Richard Preen")
set(PROJECT_CONTACT "rpreen@gmail.com")
set(PROJECT_URL "https://github.com/rpreen/xcsf")
set(PROJECT_DESCRIPTION "XCSF: Learning Classifier System")
set(PROJECT_VERSION "1.1.4")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

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

option(GEN_PROF "Generate profiling information" OFF)
if(GEN_PROF)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fprofile-generate")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate")
endif()

option(USE_PROF "Use profiling information" OFF)
if(USE_PROF)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-use")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-correction")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-correction")
endif()

option(PARALLEL "Parallel match set and prediction" ON)
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

set(CMAKE_C_FLAGS_RELEASE "-Ofast -march=native")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -flto")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -funroll-loops")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")

if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffat-lto-objects")
  set(CMAKE_C_FLAGS_RELEASE
      "${CMAKE_C_FLAGS_RELEASE} -fno-semantic-interposition")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffat-lto-objects")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
endif()

option(XCSF_MAIN "Build XCSF stand-alone main executable" ON)
if(XCSF_MAIN)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cfg/default.ini DESTINATION .)
endif()

option(XCSF_PYLIB "Build XCSF Python library" OFF)
if(XCSF_PYLIB)
  add_subdirectory(lib/pybind11)
  file(GLOB PYTHON_EXAMPLES "python/*.py")
  file(COPY ${PYTHON_EXAMPLES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(COPY "xcsf/utils/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/xcsf/utils/)
  file(COPY "xcsf/__init__.py" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/xcsf/)
  file(COPY "xcsf/__init__.pyi" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/xcsf/)
endif()

option(ENABLE_TESTS "Build unit tests" OFF)
if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

option(ENABLE_DOXYGEN "Enable Building XCSF Documentation" ON)
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
