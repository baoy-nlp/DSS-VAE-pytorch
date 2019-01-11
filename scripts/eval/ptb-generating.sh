#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ../..
python3 examples/vae_new.py --config_files configs/ptb.yaml --mode test_vaea --exp_name tune1
python3 examples/vae_new.py --config_files configs/ptb.yaml --mode test_vaea --exp_name tune2
python3 examples/vae_new.py --config_files configs/ptb.yaml --mode test_vaea --exp_name tune3
python3 examples/vae_new.py --config_files configs/ptb.yaml --mode test_vaea --exp_name tune4