#!/bin/bash


printf "\n========TSP========\n"
./scripts/run_tsp_exp.sh
printf "========TSP========\n"

printf "\n========ISO========\n"
./scripts/run_iso_exp.sh
printf "========ISO========\n"

printf "\n========COL========\n"
./scripts/run_col_exp.sh
printf "========COL========\n"

printf "\n========KNP========\n"
./scripts/run_knp_exp.sh
printf "========KNP========\n"

printf "\n========STP========\n"
./scripts/run_stp_exp.sh
printf "========STP========\n"
