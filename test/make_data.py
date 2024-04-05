#!/bin/py
import argparse
import numpy as np
import random
import sys

DEFAULT_FS         = 8000
DEFAULT_F          = 5
DEFAULT_SAMPLES    = 1000

DEFAULT_NUM_POINTS = 100
DEFAULT_START      = 50
DEFAULT_MIN        = 1
DEFAULT_MAX        = 100
DEFAULT_STEP       = 1
DEFAULT_FILE_NAME  = "data.csv"

def make_data_set(num_points, start, min, max, step, filename):
    data_point = start
    
    with open(filename, 'w') as f:
        f.write(str(data_point) + '\n')
        for i in range(num_points-1):
            data_point += random.randint(-1, 1) * float(step)
            f.write(str(data_point) + '\n')

def make_sin_wave(Fs, f, sample, filename):
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)
    with open(filename, 'w') as f:
        for val in y:
            f.write(str(val) + '\n')
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate n random numbers into a csv.') 
    
    parser.add_argument('-sin', action='store_true')
    parser.add_argument('-Fs', default=DEFAULT_FS, help='Number of points to generate')
    parser.add_argument('-f', default=DEFAULT_F, help='Starting value of data set')
    parser.add_argument('-sample', default=DEFAULT_SAMPLES, help='Minimum value of data set')
    
    parser.add_argument('-npoints', default=DEFAULT_NUM_POINTS, help='Number of points to generate')
    parser.add_argument('-start', default=DEFAULT_START, help='Starting value of data set')
    parser.add_argument('-min', default=DEFAULT_MIN, help='Minimum value of data set')
    parser.add_argument('-max', default=DEFAULT_MAX, help='Maximum value of data set')
    parser.add_argument('-step', default=DEFAULT_STEP, help='Incremental value of data set')
    parser.add_argument('-filename', default=DEFAULT_FILE_NAME, help='Name of the file')
    
    args = parser.parse_args()
    if(args.sin): 
        make_sin_wave(args.Fs, args.f, args.sample, args.filename)
    else:
        make_data_set(args.npoints, args.start, args.min, args.max, args.step, args.filename)