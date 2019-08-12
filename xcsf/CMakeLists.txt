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
 

if(NOT XCSF_PYLIB)
	#######################
	# Compile XCSF binary
	#######################

	file(GLOB SOURCES
		*.h
		*.c
	)
 
	add_executable(xcsf ${SOURCES})
	target_link_libraries(xcsf m)

else()
	##########################
	# Compile XCSF Python lib
	##########################
     
	file(GLOB SOURCES
		*.h
		*.c
		*.cpp
	)
     
	PYTHON_ADD_MODULE(xcsf ${SOURCES})
	file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)

endif()
