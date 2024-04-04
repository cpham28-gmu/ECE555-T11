#!/bin/py
import argparse
import random
import sys

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
        for i in range(num_points):
            data_point += random.randint(-1, 1) * step
            f.write(str(data_point) + '\n')
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate n random numbers into a csv.') 
    parser.add_argument('-npoints', default=DEFAULT_NUM_POINTS, help='Number of points to generate')
    parser.add_argument('-start', default=DEFAULT_START, help='Starting value of data set')
    parser.add_argument('-min', default=DEFAULT_MIN, help='Minimum value of data set')
    parser.add_argument('-max', default=DEFAULT_MAX, help='Maximum value of data set')
    parser.add_argument('-step', default=DEFAULT_STEP, help='Incremental value of data set')
    parser.add_argument('-filename', default=DEFAULT_FILE_NAME, help='Name of the file')
    
    args = parser.parse_args()
    make_data_set(args.npoints, args.start, args.min, args.max, args.step, args.filename)