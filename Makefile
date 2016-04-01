PROJECT := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)

CXX = g++
CXXFLAGS = -O3 \
           -std=c++11 \
           -Wall \
           -Wno-sign-compare \
           -fno-omit-frame-pointer

MULTIVERSO_DIR = $(PROJECT)/multiverso
MULTIVERSO_INC = $(MULTIVERSO_DIR)/include
MULTIVERSO_LIB = $(MULTIVERSO_DIR)/lib
THIRD_PARTY_LIB = $(MULTIVERSO_DIR)/third_party/lib

INC_FLAGS = -I$(MULTIVERSO_INC) -I$(PROJECT)/src -I$(PROJECT)/inference
LD_FLAGS  = -L$(MULTIVERSO_LIB) -lmultiverso
LD_FLAGS += -L$(THIRD_PARTY_LIB) -lzmq -lmpich -lmpl -lz -lpthread
  	  	
BASE_SRC = $(shell find $(PROJECT)/src -type f -name "*.cpp" -type f ! -name "ftrl.cpp")
BASE_OBJ = $(BASE_SRC:.cpp=.o)

FTRL_HEADERS = $(shell find $(PROJECT)/src -type f -name "*.h")
FTRL_SRC     = $(shell find $(PROJECT)/src -type f -name "*.cpp")
FTRL_OBJ = $(FTRL_SRC:.cpp=.o)

BIN_DIR = $(PROJECT)/bin
FTRL = $(BIN_DIR)/ftrl


all: path \
	 FTRL 

path: $(BIN_DIR)

$(BIN_DIR):
	mkdir -p $@

$(FTRL): $(FTRL_OBJ)
	$(CXX)  $(FTRL_OBJ) $(CXXFLAGS) $(INC_FLAGS) $(LD_FLAGS) -o $@

$(FTRL_OBJ): %.o: %.cpp $(FTRL_HEADERS) $(MULTIVERSO_INC)
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

FTRL: path $(FTRL)

clean:
	rm -rf  $(FTRL_OBJ) 

.PHONY: all path FTRL  clean
