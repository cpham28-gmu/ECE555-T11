#include <complex>
#include <iostream>
#include <vector>
#include <stdbool.h>
#include <cufft.h>

#include "peaks.h"

#define FFT_SIZE 100000
#define DEFAULT_FILE_NAME "input.csv"

vector<float> input(FFT_SIZE, 0);

/*
    Pull in the results from the included input.csv file 
*/
int create(string filename)
{
    FILE *f = fopen(filename.c_str(), "r");

    if (f == nullptr)
        return -1;

    char buff[255];
    char *end;

    int i;
    for (i = 0; i < FFT_SIZE; i++)
    {
        if (fgets(buff, 255, (FILE *)f) == NULL)
            break;
        input[i] = strtod(buff, &end);
    }
    fclose(f);

    return i;
}

/*
    CUDA kernel for applying scale to the results of the Forward FFT,
    Normalizes the results before magnitude and peak analysis

    Treid to use this kernel, but it was faster to scale in the main loop without a kernel
*/
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
    bool debug = false;
    cudaEvent_t time1, time2, time3, time4;
    float totalTime, ForFFTTime, ScalMagTime, RevFFTTime;
    int bins = FFT_SIZE/2 + 1;

    vector<float> data_mag_sq(bins, 0);
    vector<complex<float>> output(bins);

    // Take in file name as arg
    string filename = DEFAULT_FILE_NAME;

    // Pull entries from data points
    int count = create(filename);

    cout << "Took in " << count << " entries" << endl;

    if (debug == 1){
        printf("Input array:\n");
        for (auto &i : input) {
            printf("%f\n", i);
        }
        printf("======================\n");
    }

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

    cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

    cudaEventRecord(time1, 0);		// Before FFT
    // Run FFT and copy data
    cufftExecR2C(planReal, d_input, d_output);

    cudaMemcpyAsync(output.data(), d_output, sizeof(complex<float>) * output.size(), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaEventRecord(time2, 0);		// After FFT and Copy

    if (debug == 1){
        printf("Output after Forward FFT:\n");
        for (auto &i : output) {
            printf("%f + %fi\n", i.real(), i.imag());
        }
        printf("======================\n");
    }

    // Normalize the data
    // //kernel<<<blockSize, FFT_SIZE/blockSize, 0, stream>>>(d_output, 1.f/(bins-1));

    cudaMemcpyAsync(output.data(), d_output, sizeof(complex<float>) * output.size(), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Magnitude
    for (int i = 0; i < (bins); i++)
    {
        data_mag_sq[i] = sqrt(output[i].real() * output[i].real() + output[i].imag() * output[i].imag());
    }
    data_mag_sq[0] /= 2;
    data_mag_sq[bins-1] /= 2;

    // scale the data correctly
    for (int i = 0; i < bins; i++)
    {
        data_mag_sq[i] /= bins-1;
    }

    if (debug){
        printf("Values with magnitude squared:\n");
        for (auto i = 0; i < data_mag_sq.size(); i++) {
            printf("%f\n", data_mag_sq[i]);
        }
        printf("======================\n");
    }

    vector<peak> peaks(FFT_SIZE);
    find_peaks(data_mag_sq, peaks);

    cudaEventRecord(time3, 0);		// after normalization, magitude, and peaks

    // Reverse FFT
    cufftExecC2R(planComp, d_output, d_input);

    cudaMemcpyAsync(input.data(), d_input, sizeof(float) * input.size(), cudaMemcpyDeviceToHost, stream);

    if (debug == 1){
        printf("Values after inverted FFT:\n");
        for (auto &i : input) {
            printf("%f\n", i);
        }
        printf("======================\n");
    }

    cudaEventRecord(time4, 0);		// after reverse FFT

    cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&ForFFTTime, time1, time2);
	cudaEventElapsedTime(&ScalMagTime, time2, time3);
	cudaEventElapsedTime(&RevFFTTime, time3, time4);

    printf("Timings ------------------------------------------------------------------\n");
    printf("Forward FFT Time                    =%7.2f ms\n", ForFFTTime);
	printf("Scaling, Magnitude, and Peak Time   =%7.2f ms\n", ScalMagTime);
	printf("Reverse FFT Time                    =%7.2f ms\n", RevFFTTime);
	printf("Total time elapsed                  =%7.2f ms\n", totalTime);
    printf("--------------------------------------------------------------------------\n");

    // Free used resources
    cudaFree(d_input);
    cudaFree(d_output);

    cufftDestroy(planReal);
    cufftDestroy(planComp);
    cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
