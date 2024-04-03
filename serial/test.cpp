#include "kiss_fft.h"

int main(){
    kiss_fft_cfg cfg = kiss_fft_alloc( nfft ,is_inverse_fft ,0,0 );
    while(1){
    
        kiss_fft( cfg , cx_in , cx_out );
        

    }
    kiss_fft_free(cfg);
}
