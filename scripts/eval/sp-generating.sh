#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ../..
#python3 examples/vae_new.py --config_files configs/snli-process.yaml --mode test_generating --exp_name kl-BLEU
#python3 examples/vae_new.py --config_files configs/snli-process.yaml --mode test_generating --exp_name kl-ELBO
#python3 examples/vae_new.py --config_files configs/snli-process.yaml --mode test_generating --exp_name fkl-BLEU
#python3 examples/vae_new.py --config_files configs/snli-process.yaml --mode test_generating --exp_name fkl-ELBO
python3 examples/vae_new.py --config_files configs/snli-process.yaml --mode test_vaea --exp_name tune1
python3 examples/vae_new.py --config_files configs/snli-process.yaml --mode test_vaea --exp_name tune2
