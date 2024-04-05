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

void show_peaks(vector<peak> peaks)
{
    for (auto p : peaks)
    {
        if(p.frequency == 0 && p.value == 0) continue;
        printf("freq:%f, val:%f\n", p.frequency, p.value);
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

        /*
        if (peak_count + 1 >= MAX_PEAKS){
            break;
        }
        */
        // Add peak to the return vector
        peaks[peak_count].frequency = pt;
        peaks[peak_count].value = peak_val;
        peak_count++;

        float last_val = fftd[++pt];
        while (pt + 1 < nfft && fftd[pt + 1] < last_val)
        {
            last_val = fftd[++pt];
        }
    }

    show_peaks(peaks);

    //return peaks;
}