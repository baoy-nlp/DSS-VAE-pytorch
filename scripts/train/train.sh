#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ../..
python3 examples/main.py --config_files configs/model_configs/{2}.yaml --mode ${3} --exp_name ${4}