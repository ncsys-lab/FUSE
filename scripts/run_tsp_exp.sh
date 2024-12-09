#!/bin/bash

printf "\n=====TSP6 CONV=====\n"
python3 solver.py -be 0.00 -o -t 1000 tsp -n 6
printf "\n=====TSP8 CONV=====\n"
python3 solver.py -be -1.50 -o -t 1000 tsp -n 8
printf "\n=====TSP10 CONV=====\n"
python3 solver.py -be -1.50 -o -t 1000 tsp -n 10

printf "\n=====TSP6 ENC=====\n"
python3 solver.py -be 3.50 -o -t 1000 -f tsp -n 6
printf "\n=====TSP8 ENC=====\n"
python3 solver.py -be 2.50 -o -t 1000 -f tsp -n 8
printf "\n=====TSP10 ENC=====\n"
python3 solver.py -be 2.00 -o -t 1000 -f tsp -n 10
