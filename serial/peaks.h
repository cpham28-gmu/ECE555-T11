#include <vector>
using namespace std;

typedef struct peak
{
    float frequency;
    float value;
} peak;

void find_peaks(vector<float>, vector<peak>);