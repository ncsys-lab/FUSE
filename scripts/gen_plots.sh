#!/bin/bash

python3 vis.py logs/tsp_n8_conv_b0.00_-1.50_lin_s42/0x0F38D913_0FF68959.log.npz -n fig1a -ylim -1000 30000
python3 vis.py logs/tsp_n8_conv_b0.00_-1.50_lin_s42/0x0F38D913_0FF68959.log.npz -n fig1a_inset -ylim -100 10000 -xlim -100 100000
python3 vis.py logs/tsp_n8_enc_b0.00_2.50_lin_s42/0x0F38D913_0FF68959.log.npz -n fig1b -ylim -1000 30000
python3 vis.py logs/tsp_n8_enc_b0.00_2.50_lin_s42/0x0F38D913_0FF68959.log.npz -n fig1b_inset -ylim -1 100
