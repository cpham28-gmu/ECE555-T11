#include <vector>
using namespace std;

#define FS          100

typedef struct peak
{
    float bin;
    float value;
} peak;

void find_peaks(vector<float>, vector<peak>);