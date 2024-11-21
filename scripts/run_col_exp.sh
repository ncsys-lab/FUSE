#!/bin/bash

printf "\n=====COL6 CONV=====\n"
python3 solver.py -be 4.75 -o -t 1000 col -n 6
printf "\n=====COL8 CONV=====\n"
python3 solver.py -be 4.75 -o -t 1000 col -n 8
printf "\n=====COL10 CONV=====\n"
python3 solver.py -be 4.25 -o -t 1000 col -n 10

printf "\n=====COL6 ENC=====\n"
python3 solver.py -be 7.0 -o -t 1000 -f col -n 6
printf "\n=====COL8 ENC=====\n"
python3 solver.py -be 7.0 -o -t 1000 -f col -n 8
printf "\n=====COL10 ENC=====\n"
python3 solver.py -be 6.50 -o -t 1000 -f col -n 10
