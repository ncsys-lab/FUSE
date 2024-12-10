#!/bin/bash

sed_mod () {
    sed -nE "s/(Chip area for module.*)($1.*)/\2/p" $2
}

mod_areas() {
    sed_mod "enc_circuit'" $1
    sed_mod "adder_tree" $1
    sed_mod "weight_mux" $1
}

printf "\n=====TSP8 CONV=====\n"
python3 synth.py tsp -n 8

printf "\n=====TSP16 CONV=====\n"
python3 synth.py tsp -n 16

printf "\n=====TSP32 CONV=====\n"
python3 synth.py tsp -n 32

printf "\n=====TSP8 ENC=====\n"
python3 synth.py -f tsp -n 8
mod_areas synths/tsp_n8_enc/runs/tsp_n8_enc/1-yosys-synthesis/reports/stat.rpt

printf "\n=====TSP16 ENC=====\n"
python3 synth.py -f tsp -n 16
mod_areas synths/tsp_n16_enc/runs/tsp_n16_enc/1-yosys-synthesis/reports/stat.rpt

printf "\n=====TSP32 ENC=====\n"
python3 synth.py -f tsp -n 32
mod_areas synths/tsp_n32_enc/runs/tsp_n32_enc/1-yosys-synthesis/reports/stat.rpt
