#include <vector>
#include <iostream>

#include "peaks.h"

using namespace std;

float get_max(vector<float> fftd)
{
    float max = 0;
    for (auto val : fftd)
    {
        if (val > max)
            max = val;
    }
    return max;
}

float bin_to_frequency(float bin, int n){
    return bin * FS/2/(n);
}

void show_peaks(vector<peak> peaks, int n)
{
    for (auto p : peaks)
    {
        if((p.bin == 0 && p.value == 0) || p.bin == n) continue;
        printf("bin:%f, freq:%f, val:%f\n", p.bin, bin_to_frequency(p.bin, n), p.value);
    }
}

/*
    find_peaks()
    Inputs:
        vector<float> fftd - the complex, magnitude squared results from FFT
    Outputs:
        vector<float> - vector with peak frequencies and magnitudes

*/
void find_peaks(vector<float> fftd, vector<peak> peaks)
{
    int nfft = fftd.capacity();
    float peak_val = 0;

    float max_value = get_max(fftd);
    printf("Max val: %f\n", max_value);

    int pt = 0, peak_count = 0;
    while (pt < nfft)
    {
        // first the next 10% value
        while ((pt < nfft) && (fftd[pt] < (max_value / 50)))
        {
            pt++;
        }

        // keep counting until a decrease
        peak_val = fftd[pt];
        while (((pt + 1) < nfft) && (fftd[pt + 1] > peak_val))
        {
            peak_val = fftd[++pt];
        }

        // Add peak to the return vector
        peaks[peak_count].bin = pt;
        peaks[peak_count].value = peak_val/(nfft-1);
        peak_count++;

        float last_val = fftd[++pt];
        while (pt + 1 < nfft && fftd[pt + 1] < last_val)
        {
            last_val = fftd[++pt];
        }
    }

    show_peaks(peaks, nfft);
}