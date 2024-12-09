#!/bin/bash

printf "\n=====TSP8 CONV=====\n"
python3 synth.py -o tsp -n 8

printf "\n=====TSP16 CONV=====\n"
python3 synth.py -o tsp -n 16

printf "\n=====TSP32 CONV=====\n"
python3 synth.py -o tsp -n 32

printf "\n=====TSP8 ENC=====\n"
python3 synth.py -o -f tsp -n 8

printf "\n=====TSP16 ENC=====\n"
python3 synth.py -o -f tsp -n 16

printf "\n=====TSP32 ENC=====\n"
python3 synth.py -o -f tsp -n 32
