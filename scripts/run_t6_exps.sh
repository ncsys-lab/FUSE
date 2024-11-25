#!/bin/bash

printf "\n=====COL6 ENC (LOG)=====\n"
python3 solver.py -be 4.75 -o -t 1000 -f col -n 6 --logn
printf "\n=====COL8 ENC (LOG)=====\n"
python3 solver.py -be 4.00 -o -t 1000 -f col -n 8 --logn
printf "\n=====COL10 ENC (LOG)=====\n"
python3 solver.py -be 3.25 -o -t 1000 -f col -n 10 --logn

printf "\n=====COL6 ENC=====\n"
python3 solver.py -be 7.00 -o -t 1000 -f col -n 6
printf "\n=====COL8 ENC=====\n"
python3 solver.py -be 7.00 -o -t 1000 -f col -n 8
printf "\n=====COL10 ENC=====\n"
python3 solver.py -be 6.50 -o -t 1000 -f col -n 10
