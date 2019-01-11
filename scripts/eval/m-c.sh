#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ../..
python3 examples/vae_new.py --config_files configs/mh-s.yaml --mode test_control --exp_name ${2}
