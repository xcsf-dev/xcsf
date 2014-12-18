# 
#  Copyright (C) 2015 Richard Preen <rpreen@gmail.com>
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
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
CC=gcc

FLAGS=
CFLAGS=$(FLAGS) -Wall -Wextra -std=gnu11 -pipe -g
LDFLAGS=$(FLAGS)
LIB=-lm
 
OPT=1
GENPROF=0
USEPROF=0
SAM=0
NEURAL=0
GNUPLOT=0
RLS=0
QUADRATIC=0

ifeq ($(RLS),1)
	CFLAGS+= -DRLS
endif
ifeq ($(QUADRATIC),1)
	CFLAGS+= -DQUADRATIC
endif
ifeq ($(SAM),1)
	CFLAGS+= -DSELF_ADAPT_MUTATION
endif
ifeq ($(NEURAL),1)
	CFLAGS+= -DNEURAL_CONDITIONS
endif
ifeq ($(GNUPLOT),1)
	CFLAGS+= -DGNUPLOT
endif
ifeq ($(OPT),1)
	FLAGS+= -Ofast -march=native
endif
ifeq ($(GENPROF),1)
	FLAGS+= -fprofile-generate
endif
ifeq ($(USEPROF),1)
	FLAGS+= -fprofile-use
endif

INC=$(wildcard *.h)
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))

BIN=xcsf

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) -o $(BIN) $(OBJ) $(LDFLAGS) $(LIB)

$(OBJ): $(INC)

clean:
	$(RM) $(OBJ) $(BIN)

.PHONY: all clean
