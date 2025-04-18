CXX = clang++
CXXFLAGS = -Ofast -ffast-math -funroll-loops -fstrict-aliasing -flto -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -march=native
LDFLAGS = -flto -L/opt/homebrew/opt/libomp/lib -lomp -lm

OBJ_DIR   := bin
MAIN_SRC  := main.cpp
SRC_SRCS  := $(wildcard src/*.cpp)
OBJS      := $(OBJ_DIR)/main.o $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRC_SRCS)))

all: main

main: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJ_DIR)/main.o: main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) main
