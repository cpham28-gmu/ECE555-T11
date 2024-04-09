#include <complex>
#include <string>
#include <iostream>
#include <vector>

#include "kiss_fftr.h"
#include "peaks.h"

#include <chrono>
using namespace chrono;
using namespace std;

#define DEBUG_ARR_CONTENTS 0

#define DEFAULT_FILE_NAME "data.csv"
#define FFT_SIZE 100000
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

    vector<peak> peaks(nfft);

    // set up configs for foward FFTs
    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);

    vector<float> x(nfft, 0.0);                 // input vector
    vector<float> fx_cmplx_mag(bins, 0.0); // input vector
    vector<kiss_fft_cpx> fx(bins);      // output vector

    // Populate the input vector with real values
    for (int i = 0; i < nfft; i++)
    {
        x[i] = input_arr[i];
    }

#if DEBUG_ARR_CONTENTS
    printf("FFT inputs float: \n");
    dump_float(x);
#endif

    auto start = high_resolution_clock::now();
    // Perform the FFT
    kiss_fftr(cfg, &x[0], &fx[0]);
    auto stop = high_resolution_clock::now();
    auto fft_duration = duration_cast<microseconds>(stop - start);

    // Free the cfg and set up inverse
    kiss_fft_free(cfg);


#if DEBUG_ARR_CONTENTS
    printf("FFT results complex: \n");
    dump_cmplx(fx);
#endif
    start = high_resolution_clock::now();
    for (int i = 0; i < bins; i++)
    {
        fx_cmplx_mag[i] = sqrt(fx[i].r * fx[i].r + fx[i].i * fx[i].i);
    }
    fx_cmplx_mag[0] /= 2;
    fx_cmplx_mag[bins-1] /= 2;

#if DEBUG_ARR_CONTENTS
    printf("FFT results complex magnitude: \n");
    dump_float(fx_cmplx_mag);
#endif

    // Perform peak analysis
    find_peaks(fx_cmplx_mag, peaks);
    stop = high_resolution_clock::now();
    auto analysis_duration = duration_cast<microseconds>(stop - start);

    // Invert the FFT results back into the original inputs
    start = high_resolution_clock::now();
    cfg = kiss_fftr_alloc(nfft, 1, NULL, NULL);
    kiss_fftri(cfg, (kiss_fft_cpx *)&fx[0], &x[0]);

    // scale array back
    for (int i = 0; i < nfft; i++)
    {
        x[i] /= nfft;
    }

    stop = high_resolution_clock::now();
    auto ffti_duration = duration_cast<microseconds>(stop - start);

#if DEBUG_ARR_CONTENTS
    printf("\r\nFFT results inverted back: \n");
    dump_float(x);
#endif

    cout << "Timings ------------------------------------------------------------------" << endl;
    cout << "Forward FFT Time = " << float(fft_duration.count())/1000 << " ms" << endl;
    cout << "Scaling, Magnitude, and Peak Time = " << float(analysis_duration.count())/1000 << " ms" << endl;
    cout << "Reverse FFT Time = " << float(ffti_duration.count())/1000 << " ms" << endl;
    cout << "Total time elapsed = " << float(fft_duration.count())/1000 + float(analysis_duration.count())/1000 + float(ffti_duration.count())/1000 << " ms" << endl;
    cout << "--------------------------------------------------------------------------" << endl;

    kiss_fft_free(cfg);
    return 0;
}
