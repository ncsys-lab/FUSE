#!/bin/bash

printf "\n=====TSP6 CONV=====\n"
python3 solver.py -be 0 -o -t 1000 tsp -n 6
printf "\n=====TSP8 CONV=====\n"
python3 solver.py -be -1.5 -o -t 1000 tsp -n 8
printf "\n=====TSP10 CONV=====\n"
python3 solver.py -be -1.5 -o -t 1000 tsp -n 10

printf "\n=====TSP6 ENC=====\n"
python3 solver.py -be 3.5 -o -t 1000 -f tsp -n 6
printf "\n=====TSP8 ENC=====\n"
python3 solver.py -be 2.5 -o -t 1000 -f tsp -n 8
printf "\n=====TSP10 ENC=====\n"
python3 solver.py -be 2.0 -o -t 1000 -f tsp -n 10
