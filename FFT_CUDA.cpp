#include <complex>
#include <iostream>
#include <vector>
#include <cufft.h>

#include "peaks.h"

#define FFT_SIZE 8

__global__
void kernel(cufftComplex* data, float scale) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i<FFT_SIZE; i+= stride) {
        data[tid].x *= scale;
        data[tid].y *= scale;
    }
}

int main(int argc, char *argv[]) {
    cufftHandle planReal, planComp;
    cudaStream_t stream = NULL;

    vector<float> input(FFT_SIZE, 0);
    vector<float> data_mag_sq(FFT_SIZE/2, 0);
    vector<complex<float>> output(FFT_SIZE / 2 + 1);

    for (auto i = 0; i < FFT_SIZE; i++) {
        input[i] = i;
    }

    printf("Input array:\n");
    for (auto &i : input) {
        printf("%f\n", i);
    }
    printf("======================\n");

    float *d_input = nullptr;
    cufftComplex *d_output = nullptr;

    cufftCreate(&planReal);
    cufftCreate(&planComp);
    cufftPlan1d(&planReal, FFT_SIZE, CUFFT_R2C, 1);
    cufftPlan1d(&planComp, FFT_SIZE, CUFFT_C2R, 1);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(planReal, stream);
    cufftSetStream(planComp, stream);

    // Create device data arrays
    cudaMalloc((void **)&d_input, sizeof(float) * input.size());
    cudaMalloc((void **)&d_output, sizeof(complex<float>) * output.size());
    cudaMemcpyAsync(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice, stream);

    // Run FFT and copy data
    cufftExecR2C(planReal, d_input, d_output);

    cudaMemcpyAsync(output.data(), d_output, sizeof(complex<float>) * output.size(), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    printf("Output after Forward FFT:\n");
    for (auto &i : output) {
        printf("%f + %fi\n", i.real(), i.imag());
    }
    printf("======================\n");

    // Normalize the data
    //kernel<<<1, 128, 0, stream>>>(d_output, 1.f/FFT_SIZE);

    // Magnitude
    for (int i = 0; i < (FFT_SIZE/2); i++)
    {
        data_mag_sq[i] = output[i].real() * output[i].real() + output[i].imag() * output[i].imag();
    }
    data_mag_sq[0] /= 2;
    data_mag_sq[(FFT_SIZE/2)-1] /= 2;

    printf("Values with magnitude squared:\n");
    for (auto i = 0; i < input.size()/2; i++) {
        printf("%f\n", data_mag_sq[i]);
    }
    printf("======================\n");

    vector<peak> peaks(FFT_SIZE);
    find_peaks(data_mag_sq, peaks);

    // Reverse FFT
    cufftExecC2R(planComp, d_output, d_input);

    cudaMemcpyAsync(output.data(), d_input, sizeof(float) * input.size(), cudaMemcpyDeviceToHost, stream);

    printf("Values after inverted FFT:\n");
    for (auto &i : input) {
        printf("%f\n", i);
    }
    printf("======================\n");

    // Free used resources
    cudaFree(d_input);
    cudaFree(d_output);

    cufftDestroy(planReal);
    cufftDestroy(planComp);
    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}