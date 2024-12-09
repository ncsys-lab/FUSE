#!/bin/bash

printf "\n=====ISO6 ENC=====\n"
python3 solver.py -be 2.25 -o -t 1000 -f iso -n 6
printf "\n=====ISO8 ENC=====\n"
python3 solver.py -be 1.25 -o -t 1000 -f iso -n 8
printf "\n=====ISO10 ENC=====\n"
python3 solver.py -be 0.50 -o -t 1000 -f iso -n 10

printf "\n=====ISO6 CONV=====\n"
python3 solver.py -be 0.75 -o -t 1000 iso -n 6
printf "\n=====ISO8 CONV=====\n"
python3 solver.py -be 0.00 -o -t 1000 iso -n 8
printf "\n=====ISO10 CONV=====\n"
python3 solver.py -be -0.25 -o -t 1000 iso -n 10
