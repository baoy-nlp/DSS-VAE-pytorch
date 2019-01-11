# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append(".")
from utils.config_utils import dict_to_args
from utils.config_utils import yaml_load_dict
from preprocess.nmt_process import make_nmt_dataset
from preprocess.nmt_process import make_nmt_simple_dataset
from preprocess.generate_dataset import prepare_dataset


def nmt_preprocess():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/nmt.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    make_nmt_dataset(
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        tgt_dir=args.origin_tgts,
    )


def nmt_construction():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/nmt.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=data_dir,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=-1,
        max_tgt_length=-1,
    )
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=args.sample_tgts,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=args.max_src_length,
        max_tgt_length=args.max_tgt_length,
    )


def mt_preprocess():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/mt.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    make_nmt_simple_dataset(
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        tgt_dir=args.origin_tgts,
    )


def mt_construction():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/mt.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=data_dir,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=-1,
        max_tgt_length=-1,
    )


def webnlg_preprocess():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/webnlg.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    make_nmt_simple_dataset(
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        tgt_dir=args.origin_tgts,
    )


def webnlg_construction():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/webnlg.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=data_dir,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=-1,
        max_tgt_length=-1,
    )


def webnlg_robust_construction():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/webnlg.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=args.sample_tgts,
        max_src_vocab=6000,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=-1,
        max_tgt_length=-1,
    )


def s2b_construction(is_write=True):
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/ptb.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=data_dir,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=-1,
        max_tgt_length=-1,
        write_down=is_write
    )


def snli_process_construction(is_write=True):
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/snli-process.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=args.target_tgts,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=args.max_src_length,
        max_tgt_length=args.max_tgt_length,
        write_down=is_write,
    )


def snli_sample_construction(is_write=True):
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/snli-sample.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=args.target_tgts,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=-1,
        max_tgt_length=-1,
        write_down=is_write,
    )


def quora_construction(config_file="/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora-50k.yaml", is_write=True):
    # config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora-50k.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dir = args.origin_tgts
    data_dict = {
        "train": "train.s2b",
        "dev": "dev.s2b",
        "test": "test.s2b",
    }
    prepare_dataset(
        data_dir=data_dir,
        data_dict=data_dict,
        tgt_dir=args.data_tgts,
        max_src_vocab=args.max_src_vocab,
        max_tgt_vocab=args.max_tgt_vocab,
        vocab_freq_cutoff=args.cut_off,
        max_src_length=args.max_src_length,
        max_tgt_length=args.max_tgt_length,
        train_size=args.train_size,
        write_down=is_write
    )


if __name__ == '__main__':
    # nmt_construction()
    # mt_construction()
    # webnlg_preprocess()
    # webnlg_construction()
    # webnlg_robust_construction()
    # s2b_construction(False)
    # snli_process_construction(False)
    # snli_sample_construction(False)
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora.yaml"
    quora_construction(config_file=config_file, is_write=True)
