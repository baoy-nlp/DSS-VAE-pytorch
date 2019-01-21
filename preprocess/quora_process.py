# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(".")
from utils.config_utils import dict_to_args
from utils.config_utils import yaml_load_dict
from struct_self.phrase_tree import PhraseTree
from utils.utility import write_docs
from preprocess.tree_convert import make_s2b_dataset
import random


def parsed_to_pair(args=None):
    if args is None:
        config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora.yaml"
        args_dict = yaml_load_dict(config_file)
        args = dict_to_args(args_dict)
    origin_file = os.path.join(args.origin_tgts, args.origin_file)
    label_file = os.path.join(args.origin_tgts, args.label_file)
    pair_file = os.path.join(args.origin_tgts, args.pair_file)
    unpair_file = os.path.join(args.origin_tgts, args.unpair_file)

    with open(origin_file, "r") as f:
        tree_list = [line.strip() for line in f.readlines()]

    with open(label_file, "r") as f:
        label_list = [line.strip().split(" ") for line in f.readlines()]

    pair_list = []
    unpair_list = []
    for label in label_list:
        try:
            num_i1 = int(label[0]) - 1
            t1 = tree_list[num_i1]
            num_i2 = int(label[1]) - 1
            t2 = tree_list[num_i2]
            num_l = int(label[2])

            if len(t1.strip()) > 0 and len(t2.strip()) > 0:
                item = [t1, t2]
                item_str = "\t".join(item)
                if num_l == 0:
                    unpair_list.append(item_str)
                elif num_l == 1:
                    pair_list.append(item_str)
            else:
                print(t1)
                print(t2)
        except:
            pass

    with open(pair_file, "w") as f:
        for line in pair_list:
            f.write(line.strip())
            f.write("\n")
    with open(unpair_file, "w") as f:
        for line in unpair_list:
            f.write(line.strip())
            f.write("\n")


def load_tree_file(tree_file):
    tree_list = []
    with open(tree_file, "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            tree_list.append(PhraseTree.parse(line[0]))
            tree_list.append(PhraseTree.parse(line[1]))
    return tree_list


def load_to_pair_tree(tree_file):
    tree_list = []
    with open(tree_file, "r") as f:
        for line in f.readlines():
            try:
                line = line.strip().split("\t")
                itree = PhraseTree.parse(line[0])
                otree = PhraseTree.parse(line[1])
                tree_list.append((itree, otree))
            except:
                print(line)
    return tree_list


def prepare_con():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    pair_file = os.path.join(args.origin_tgts, args.pair_file)
    unpair_file = os.path.join(args.origin_tgts, args.unpair_file)
    pair_tree = load_tree_file(pair_file)
    unpair_tree = load_tree_file(unpair_file)
    all_size = args.train_size + args.valid_size + args.test_size
    all_tree = pair_tree + unpair_tree

    all_idx = random.sample(range(len(all_tree)), all_size)
    train_idx = all_idx[:args.train_size]
    valid_idx = all_idx[args.train_size:args.train_size + args.valid_size]
    test_idx = all_idx[-args.test_size:]

    train_tree = [all_tree[idx] for idx in train_idx]
    valid_tree = [all_tree[idx] for idx in valid_idx]
    test_tree = [all_tree[idx] for idx in test_idx]
    if not os.path.exists(args.data_tgts):
        os.makedirs(args.data_tgts)
    write_docs(docs=train_tree, fname=os.path.join(args.data_tgts, "train.con"))
    write_docs(docs=valid_tree, fname=os.path.join(args.data_tgts, "dev.con"))
    write_docs(docs=test_tree, fname=os.path.join(args.data_tgts, "test.con"))


def prepare_s2b(config_file="/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora.yaml"):
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    make_s2b_dataset(
        train_file=os.path.join(args.data_tgts, "train.con"),
        dev_file=os.path.join(args.data_tgts, "dev.con"),
        test_file=os.path.join(args.data_tgts, "test.con"),
        tgt_dir=args.data_tgts
    )


def prepare_paraphrase():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    pair_file = os.path.join(args.origin_tgts, args.pair_file)
    pair_trees = load_to_pair_tree(pair_file)

    select_idx = random.sample(range(len(pair_trees)), args.test_size)
    test_paraphrase = []
    for idx in select_idx:
        pair_tree = pair_trees[idx]
        test_paraphrase.append(
            "\t".join([pair_tree[0].words, pair_tree[1].words])
        )
    write_docs(fname=os.path.join(args.origin_tgts, "para.text"), docs=test_paraphrase)


def prepare_raw_paraphrase():
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/quora.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    with open(os.path.join(args.origin_tgts, "para.raw.token"), "r") as f:
        pair_raws = [line for line in f.readlines()]

    select_idx = random.sample(range(len(pair_raws)), args.test_size)
    test_paraphrase = []
    for idx in select_idx:
        test_paraphrase.append(pair_raws[idx].strip())
    write_docs(fname=os.path.join(args.origin_tgts, "para.raw.text"), docs=test_paraphrase)


file = "/home/user_data/baoy/projects/seq2seq_parser/configs/data_configs/q50s.yaml"
# parsed_to_pair()
# prepare_con()
prepare_s2b(config_file=file)
# prepare_paraphrase()
# prepare_raw_paraphrase()
