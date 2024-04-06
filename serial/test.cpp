#include <complex>
#include <string>
#include <iostream>
#include <vector>

#include "kiss_fftr.h"
#include "peaks.h"

using namespace std;

#define DEBUG_ARR_CONTENTS 1

#define DEFAULT_FILE_NAME "data.csv"
#define FFT_SIZE 1024
#define MIN_ENTRIES 10

float input_arr[FFT_SIZE];

void dump_float(vector<float> arr)
{
    for (auto i : arr)
        cout << i << ' ';
    cout << endl
         << endl;
}

void dump_cmplx(vector<kiss_fft_cpx> arr)
{
    for (auto i : arr)
        cout << i.r << "+(" << i.i << ")i ";
    cout << endl
         << endl;
}

/*
    create() - reads in the csv data from the file and stores it in memory
    Inputs:
        string filename - the name of the file to be read
    Outputs:
        int - number of data_points read

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
        input_arr[i] = strtof(buff, &end);
        // printf("%lld\n", NumbersO[i]);
    }
    fclose(f);

    return i;
}

int main(int argc, char *argv[])
{
    // Take in file name as arg
    string filename = DEFAULT_FILE_NAME;
    if (argc > 1)
    {
        filename = argv[1];
    }

    // Pull entries from data points
    int count = create(filename);

    cout << "Took in " << count << " entries" << endl;

    // don't run
    if (count <= MIN_ENTRIES)
    {
        cout << "Not enough data to process! Exiting..." << endl;
        return -1;
    }

    // subtract one if odd since kissfft needs even numbers
    int nfft = count - (count % 2 == 1);
    int bins = nfft / 2 + 1;

    // set up configs for foward FFTs
    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);

    vector<float> x(nfft, 0.0);                 // input vector
    vector<float> fx_mag_sq(bins, 0.0); // input vector
    vector<kiss_fft_cpx> fx(bins);      // output vector

    float alpha = 25.0/46;
	float factor = 2.0 * M_PI / nfft;

    // Populate the input vector with real values
    for (int i = 0; i < nfft; i++)
    {
        x[i] = input_arr[i];
    }

#if DEBUG_ARR_CONTENTS
    printf("FFT inputs float: \n");
    dump_float(x);
#endif

    // Populate the input vector with real values
    for (int i = 0; i < nfft; i++)
    {
        float f  = alpha - (1.0 - alpha) * cos (factor * i);
        x[i] *= f;
    }

    #if DEBUG_ARR_CONTENTS
    printf("FFT inputs hamming windowed: \n");
    dump_float(x);
#endif

    // Perform the FFT
    kiss_fftr(cfg, &x[0], &fx[0]);

    // Free the cfg and set up inverse
    kiss_fft_free(cfg);
    cfg = kiss_fftr_alloc(nfft, 1, NULL, NULL);

#if DEBUG_ARR_CONTENTS
    printf("FFT results complex: \n");
    dump_cmplx(fx);
#endif

    for (int i = 0; i < bins; i++)
    {
        fx_mag_sq[i] = fx[i].r * fx[i].r + fx[i].i * fx[i].i;
    }
    fx_mag_sq[0] /= 2;
    fx_mag_sq[bins-1] /= 2;

#if DEBUG_ARR_CONTENTS
    printf("FFT results magnitude squared: \n");
    dump_float(fx_mag_sq);
#endif

    // Perform peak analysis
    vector<peak> peaks(nfft);
    find_peaks(fx_mag_sq, peaks);

    // Invert the FFT results back into the original inputs
    kiss_fftri(cfg, (kiss_fft_cpx *)&fx[0], &x[0]);

    // scale array back
    for (int i = 0; i < nfft; i++)
    {
        float f  = alpha - (1.0 - alpha) * cos (factor * i);
        x[i] /= nfft * f;
    }

#if DEBUG_ARR_CONTENTS
    printf("\r\nFFT results inverted back: \n");
    dump_float(x);
#endif

    kiss_fft_free(cfg);
    return 0;
}
