#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ../..
python3 examples/vae_new.py --config_files configs/q500s.yaml --mode ${2} --exp_name ${3}