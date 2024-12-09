#!/bin/bash

printf "\n=====STP10 CONV=====\n"
python3 solver.py -be 0.0 -o -t 1000 stp -n 10 -u 5
printf "\n=====STP15 CONV=====\n"
python3 solver.py -be 0.0 -o -t 1000 stp -n 15 -u 8
printf "\n=====STP20 CONV=====\n"
python3 solver.py -be 0.0 -o -t 1000 stp -n 20 -u 10

printf "\n=====STP10 ENC=====\n"
python3 solver.py -be 3.50 -o -t 1000 -f stp -n 10 -u 5
printf "\n=====STP15 ENC=====\n"
python3 solver.py -be 3.25 -o -t 1000 -f stp -n 15 -u 8
printf "\n=====STP20 ENC=====\n"
python3 solver.py -be 3.00 -o -t 1000 -f stp -n 20 -u 10
