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
#  Building Python library REQUIRES: 
#  libboost-python-dev

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

option(PARALLEL "Parallel match set and prediction" ON)

if(PARALLEL)
	find_package(OpenMP REQUIRED)
	add_definitions(${OpenMP_C_FLAGS})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_MATCH")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPARALLEL_PRED")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	add_definitions(${OpenMP_CXX_FLAGS})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()
 
option(XCSF_PYTHON_LIBRARY "Build XCSF Python library" OFF)

if(NOT XCSF_PYTHON_LIBRARY)
	    #######################
		# Compile XCSF binary
		#######################
	          
		option(GNUPLOT "Plot performance with GNUPLOT" ON)
		option(GENPROF "Generate profile" OFF)
		option(USEPROF "Use profile" OFF)
	 
		if(GNUPLOT)
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGNUPLOT")
		endif()
	 	
		if(GENPROF)
			set(PROFILE_FLAGS "-fprofile-generate")
		endif()
			
		if(USEPROF)
			set(PROFILE_FLAGS "-fprofile-use")
		endif()
	 
		add_subdirectory(xcsf)
		file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xcsf/out)
	 
else()
	##########################
	# Compile XCSF Python lib
	##########################

	FIND_PACKAGE(PythonInterp)

	if (PYTHONINTERP_FOUND)
		if (UNIX AND NOT APPLE)
			if (PYTHON_VERSION_MAJOR EQUAL 3)
				FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_SUFFIX})
				FIND_PACKAGE(PythonInterp 3)
				FIND_PACKAGE(PythonLibs 3 REQUIRED)
			else()
				FIND_PACKAGE(Boost COMPONENTS python)
				FIND_PACKAGE(PythonInterp)
				FIND_PACKAGE(PythonLibs REQUIRED)
			endif()
		else()
			if (PYTHON_VERSION_MAJOR EQUAL 3)
				FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
				FIND_PACKAGE(PythonInterp 3)
				FIND_PACKAGE(PythonLibs 3 REQUIRED)
			else()
				FIND_PACKAGE(Boost COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
				FIND_PACKAGE(PythonInterp)
				FIND_PACKAGE(PythonLibs REQUIRED)
			endif()
		endif()
	else()
		message("Python not found")
	endif()

	message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
	message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
	message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
	message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")

	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
	LINK_LIBRARIES(${Boost_LIBRARIES} boost_numpy ${PYTHON_LIBRARIES})

	add_subdirectory(xcsf)

	file(COPY MyProject.py DESTINATION .)             
	file(COPY xcsf/cons.txt DESTINATION .)             

endif()

message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
