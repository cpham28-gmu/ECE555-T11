clear;

Fs = 100;            % Sampling frequency, e.g., Hz (samples/sec)
T = 1/Fs;             % Sampling period (time between samples, e.g., sec)
L = 1000;             % Length of signal (in samples)
t = (0:L-1)*T;        % Time vector

% Sine wave parameters
theta = 0;            % Phase, in radians

F = 1;               % Frequency of sin wave (cycles/sec)
x1 = sin(2 * pi * F * t + theta);
F = 2;
x2 = sin(2 * pi * F * t + theta);
F = 3;
x3 = sin(2 * pi * F * t + theta);


x = (x1 + x2 + x3);

csvwrite("sin3.csv", permute(x,[2,1]))

#hamming window, permuted to match input signals dimensions
h = permute(hamming(length(t)), [2,1]);

#comment hamming for now
#multiply input signal by hamming window and take FFT
#for i = 1:length(t);
#  z(i) = h(i)*x(i);
#endfor;
fft_var = fft(x);
fft_var = 2 * fft_var(1:length(t)/2+1);

#compute complex magnitudes of frequencies
fft_cmplx_mag = abs(fft_var/length(t));

max_value = max(fft_cmplx_mag); #find max value of FFT

#arrays to hold peak information
peaks_bin = [];
peaks_val = [];

pt = 1;
nfft = length(fft_cmplx_mag);

#iterate through the array
while pt < nfft;

  #iterate until we reach an increase surpassing a threshold value
  while(pt < nfft && fft_cmplx_mag(pt) < max_value/50);
    pt++;
  endwhile;

  #iterate until we find a decrease
  pk_v = fft_cmplx_mag(pt);
  while(pt+2 < nfft + 2 && fft_cmplx_mag(pt+1) > pk_v)
    pk_v = fft_cmplx_mag(pt);
    pt++;
  endwhile;

  #record the peak
  peaks_bin(length(peaks_bin)+1) = pt;    #frequency
  peaks_val(length(peaks_val)+1) = x(pt); #magnitude

  #iterate until we find the next increase
  last_v = fft_cmplx_mag(pt);
  pt++;
  while(pt+2 < nfft + 1 && fft_cmplx_mag(pt+1) < last_v);
    last_v = fft_cmplx_mag(pt);
    pt++;
  endwhile;

endwhile

peaks_bin *= Fs /2 / (nfft+1)
peaks_val

#plot input signal with peaks
hold on
#plot(peaks_bin, peaks_val,'o');

f = Fs*(0:(length(t)/2))/length(t);
#plot( fft_cmplx_mag(1:100));

plot(x);
#plot(x1);
#plot(x2);
#plot(x3);