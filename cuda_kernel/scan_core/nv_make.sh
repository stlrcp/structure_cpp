#! /bin/bash

nvcc -std=c++11 -g -o cupy_scan -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart cupy_scan.cu
