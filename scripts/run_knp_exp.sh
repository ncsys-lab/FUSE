#!/bin/bash

printf "\n=====KNP10 CONV=====\n"
python3 solver.py -be -2.5 -o -t 1000 knp -n 10
printf "\n=====KNP20 CONV=====\n"
python3 solver.py -be -3.0 -o -t 1000 knp -n 20
printf "\n=====KNP30 CONV=====\n"
python3 solver.py -be -3.0 -o -t 1000 knp -n 30

printf "\n=====KNP10 ENC=====\n"
python3 solver.py -be 2.5 -o -t 1000 -f knp -n 10
printf "\n=====KNP20 ENC=====\n"
python3 solver.py -be 1.5 -o -t 1000 -f knp -n 20
printf "\n=====KNP30 ENC=====\n"
python3 solver.py -be 1.0 -o -t 1000 -f knp -n 30
