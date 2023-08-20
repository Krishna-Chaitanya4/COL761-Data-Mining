#!/bin/bash
CPP_FILE_1="compressing.cpp"
CPP_FILE_2="decompressing.cpp"
OUTPUT_EXECUTABLE_1="a1"
OUTPUT_EXECUTABLE_2="a2"
g++ -O2 "$CPP_FILE_1" -o "$OUTPUT_EXECUTABLE_1"
g++ -O2 "$CPP_FILE_2" -o "$OUTPUT_EXECUTABLE_2"
