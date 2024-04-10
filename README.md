# ECE555-T11

### Uses cuFFT library with CUDA to perform peak analysis

### Setup

create a directory to store the Makefile, FFT_CUDA.cpp, peaks.cpp, peaks.h, and input.csv files.

code can be comiled and run using nvcc using the following commands in terminal

```
make FFT_CUDA
./FFT_CUDA
```

the FFT_SIZE can be changed by editing the FFT_SZIE in the top of the file **note that it needs to be a power of 2**
