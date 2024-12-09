#!/bin/bash

printf "\n=====TSP100 ENC=====\n"
python3 solver.py -i 5000000 -q 5000 -be 2.25 -o -t 1000 -f tsp -n 100

printf "\n=====ISO18 ENC=====\n"
python3 solver.py -i 5000000 -q 5000 -be 0.00 -o -t 1000 -f iso -n 18

printf "\n=====COL50 ENC=====\n"
python3 solver.py -i 5000000 -q 5000 -be 2.50 -o -t 1000 -f col -n 50

printf "\n=====KNP100 ENC=====\n"
python3 solver.py -i 5000000 -q 5000 -be 0.25 -o -t 1000 -f knp -n 100

printf "\n=====STP100 ENC=====\n"
python3 solver.py -i 5000000 -q 5000 -be 3.75 -o -t 1000 -f stp -n 100 -u 50
