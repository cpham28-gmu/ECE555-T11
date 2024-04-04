#include "kiss_fft.h"
#include <complex>
#include <string>
#include <iostream>
#include <vector>
#include <kiss_fftr.h>

using namespace std;

#define DEFAULT_FILE_NAME "data.csv"
#define FFT_SIZE 1024

float input_arr[FFT_SIZE];

void dump_float(float *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("\t%f\n", arr[i]);
    }
}

void dump_cmplx(kiss_fft_cpx *arr, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("\t%f:%f\n", arr[i].r / n, arr[i].i);
    }
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

    return i;
}

int main(int argc, char *argv[])
{
    string filename = DEFAULT_FILE_NAME;
    if (argc > 1)
    {
        filename = argv[1];
    }

    int count = create(filename);

    cout << "Took in " << count << " entries" << endl;

    if (count <= 0)
    {
        cout << "Nothing to process! Exiting..." << endl;
        return -1;
    }

    // subtract one if odd since kissfft needs even numbers
    int nfft = count - (count % 2 == 1);

    kiss_fftr_cfg fwd = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    kiss_fftr_cfg inv = kiss_fftr_alloc(nfft, 1, NULL, NULL);

    vector<float> x(nfft, 0.0);
    vector<kiss_fft_cpx> fx(nfft/2 + 1);

    for (int i = 0; i < nfft; ++i)
    {
        x[i] = input_arr[i];
    }

    cout << x.capacity() << endl;
    for (float i : x)
        cout << i << ' ';
    cout << endl << endl;

    kiss_fftr(fwd, &x[0], &fx[0]);

    for (kiss_fft_cpx i : fx)
        cout << i.r << ':' << i.i << ' ';
    cout << endl << endl;

    kiss_fftri(inv, (kiss_fft_cpx *)&fx[0], &x[0]);

    for (float i : x)
        cout << i << ' ';
    cout << endl << endl;

    kiss_fft_free(fwd);
    kiss_fft_free(inv);
    return 0;
}
