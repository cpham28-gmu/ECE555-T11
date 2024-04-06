CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I../utils
LIBS         := -L$(CUDA_TOOLKIT)/lib64 -lcufft
FLAGS        := -O3 -std=c++11

all: FFT_CUDA

FFT_CUDA: *.cpp 
	nvcc -x cu $(FLAGS) $(INC) *.cpp -o FFT_CUDA $(LIBS)