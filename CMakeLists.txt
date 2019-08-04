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
#

cmake_minimum_required(VERSION 2.8.3)
 
project(XCSF CXX C)
set(PROJECT_VENDOR "Richard Preen")
set(PROJECT_CONTACT "rpreen@gmail.com")
set(PROJECT_URL "https://github.com/rpreen/xcsf")
set(PROJECT_DESCRIPTION "XCSF: Learning Classifier System")
 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -Wall -Wextra ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunused ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmaybe-uninitialized")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfatal-errors")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-qual")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wredundant-decls")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winit-self")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -march=native")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-Ofast -march=native -g3 ") 

option(GNUPLOT "Plot performance with GNUPLOT" ON)
option(PARALLEL "Parallel match set and prediction" ON)
option(GENPROF "Generate profile" OFF)
option(USEPROF "Use profile" OFF)

if(GNUPLOT)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGNUPLOT")
endif()

if(PARALLEL)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_MATCH")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_PRED")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif()
	
if(GENPROF)
	set(PROFILE_FLAGS "-fprofile-generate")
endif()
	
if(USEPROF)
	set(PROFILE_FLAGS "-fprofile-use")
endif()

message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
 
add_subdirectory(xcsf)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xcsf/out)
