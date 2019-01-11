#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ../..
python3 examples/vae_new.py --config_files configs/snli-process.yaml --mode test_vaea --exp_name ${2}