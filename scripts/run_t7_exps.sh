#!/bin/bash

printf "\n=====TSP100 ENC=====\n"
python3 solver.py -i 5000000 -q 5000 -be 3.75 -o -t 1000 -f tsp -n 100
