CXX      := clang++
CXXFLAGS := -O3 -g -Werror -std=c++17 -fopenmp
LDFLAGS  := -lm


OBJ_DIR   := bin
MAIN_SRC  := main.cpp
SRC_SRCS  := $(wildcard src/*.cpp)


OBJS      := $(OBJ_DIR)/main.o $(patsubst src/%.cpp, $(OBJ_DIR)/%.o, $(notdir $(SRC_SRCS)))


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
