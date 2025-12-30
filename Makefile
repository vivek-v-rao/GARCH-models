# Makefile for xgarch_data
#
# usage:
#   make
#   make run
#   make clean

CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS :=

EXE := xgarch_data.exe

SRC := xgarch_data.cpp
SRC += dataframe.cpp
SRC += date_utils.cpp
SRC += util.cpp
SRC += cli.cpp
SRC += dist.cpp
SRC += param_maps.cpp
SRC += vol_models.cpp
SRC += nelder_mead.cpp
SRC += fit_nagarch.cpp
SRC += fit_garch.cpp
SRC += simulate.cpp
SRC += report.cpp
SRC += stats.cpp
SRC += xgarch_utils.cpp

OBJ := $(SRC:.cpp=.o)

.PHONY: all run clean

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE) $(OBJ)
