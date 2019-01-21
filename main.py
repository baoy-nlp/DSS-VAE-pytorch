from __future__ import absolute_import

import argparse

from examples.test import test_vae
from examples.train import train_vae, train_ae, train_nae
from utils.config_utils import dict_to_args
from utils.config_utils import yaml_load_dict


def process_args():
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--config_files', type=str, help='config_files')
    opt_parser.add_argument('--exp_name', type=str, help='config_files')
    opt_parser.add_argument('--load_src_lm', type=str, default=None)
    opt_parser.add_argument('--mode', type=str, default=None)
    opt = opt_parser.parse_args()

    configs = yaml_load_dict(opt.config_files)

    base_args = dict_to_args(configs['base_configs']) if 'base_configs' in configs else None
    baseline_args = dict_to_args(configs['baseline_configs']) if 'baseline_configs' in configs else None
    prior_args = dict_to_args(configs['prior_configs']) if 'prior_configs' in configs else None
    encoder_args = dict_to_args(configs['encoder_configs']) if 'encoder_configs' in configs else None
    decoder_args = dict_to_args(configs['decoder_configs']) if 'decoder_configs' in configs else None
    vae_args = dict_to_args(configs['vae_configs']) if 'vae_configs' in configs else None
    ae_args = dict_to_args(configs["ae_configs"]) if 'ae_configs' in configs else None
    nae_args = dict_to_args(configs["nag_configs"]) if 'nag_configs' in configs else None

    if base_args is not None:
        if opt.mode is not None:
            base_args.mode = opt.mode
        if opt.exp_name is not None:
            base_args.exp_name = opt.exp_name
        if opt.load_src_lm is not None:
            base_args.load_src_lm = opt.load_src_lm

    return {
        'base': base_args,
        "baseline": baseline_args,
        'prior': prior_args,
        'encoder': encoder_args,
        "decoder": decoder_args,
        "vae": vae_args,
        "ae": ae_args,
        "nae": nae_args
    }


if __name__ == "__main__":
    config_args = process_args()
    args = config_args['base']
    if args.mode == "train_vae":
        train_vae(args, config_args['vae'])
    elif args.mode == "train_ae":
        train_ae(args, config_args['ae'])
    elif args.mode == "train_nae":
        train_nae(args, config_args['nae'])
    elif args.mode == "test_vae":
        raw_sent = int(input("select test mode: "))
        test_vae(args, config_args['vae'], input_mode=raw_sent)
    elif args.mode == "test_vaea":
        test_vae(args, config_args['vae'], input_mode=0)
    elif args.mode == "test_generating":
        test_vae(args, config_args['vae'], input_mode=1)
    elif args.mode == "test_paraphrase":
        test_vae(args, config_args['vae'], input_mode=2)
    elif args.mode == "test_control":
        test_vae(args, config_args['vae'], input_mode=3)
    elif args.mode == "test_transfer":
        test_vae(args, config_args['vae'], input_mode=4)
    elif args.mode == "test_pure_para":
        test_vae(args, config_args['vae'], input_mode=5)
    else:
        raise NotImplementedError
