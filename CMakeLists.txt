# 
#  Copyright (C) 2019 Richard Preen <rpreen@gmail.com>
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

cmake_minimum_required(VERSION 3.12)

project(XCSF CXX C)
set(PROJECT_VENDOR "Richard Preen")
set(PROJECT_CONTACT "rpreen@gmail.com")
set(PROJECT_URL "https://github.com/rpreen/xcsf")
set(PROJECT_DESCRIPTION "XCSF: Learning Classifier System")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -Wall -Wextra ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfatal-errors")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-qual")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wredundant-decls")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winit-self")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe")
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wuninitialized")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmaybe-uninitialized")
endif()
 
option(GNUPLOT "Plot performance with GNUPLOT" OFF)
if(GNUPLOT)
	find_package(Gnuplot REQUIRED)
	if(GNUPLOT_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGNUPLOT")
		file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/out)
	endif()
endif()
 
option(PARALLEL "Parallel match set and prediction" ON)
if(PARALLEL)
	find_package(OpenMP REQUIRED)
	if(OpenMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_MATCH -DPARALLEL_PRED")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPARALLEL_MATCH -DPARALLEL_PRED")
	endif()
endif()

option(XCSF_PYLIB "Build XCSF Python library" OFF)
if(XCSF_PYLIB)
	find_package(Python3)
	if(OSX OR !Python3_FOUND)
		find_package(Boost COMPONENTS python numpy REQUIRED)
		find_package(PythonInterp REQUIRED)
		find_package(PythonLibs REQUIRED)
	else()
		find_package(Boost COMPONENTS python3 numpy3 REQUIRED)
		find_package(PythonInterp 3 REQUIRED)
		find_package(PythonLibs 3 REQUIRED)
	endif()
	include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
	link_libraries(${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
	file(COPY MyProject.py DESTINATION .)             
	message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
	message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
	message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
	message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
endif()

add_subdirectory(xcsf)
file(COPY default.ini DESTINATION .)             
 
set(CMAKE_C_FLAGS_DEBUG "-g3")
set(CMAKE_C_FLAGS_RELEASE "-Ofast -march=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -flto")

message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
