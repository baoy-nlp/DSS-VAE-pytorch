# coding=utf-8
"""
convert the ptb to linearized tree
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

sys.path.append(".")

from preprocess.tree_linearization import tree_convert
from preprocess.tree_linearization import tree_to_s2b


def write_docs(fname, docs):
    with open(fname, 'w') as f:
        for doc in docs:
            f.write(str(doc))
            f.write('\n')


def remove_same(docs):
    check = {}
    res = []
    for doc in docs:
        if doc not in check:
            check[doc] = 1
            res.append(doc)
        else:
            pass
    print("same data filter:{}".format(len(docs) - len(res)))
    return res


def ptb_to_s2b(tree_file, rm_same=False):
    with open(tree_file, 'r') as tf:
        s2b_list = []
        for tree_str in tf.readlines():
            if len(tree_str.strip()) > 0:
                if not tree_str.strip().startswith("(TOP"):
                    tree_str = "(TOP " + tree_str.strip() + ")"
                try:
                    src, tgt = tree_to_s2b(tree_str.strip())
                    s2b_list.append("\t".join([src, tgt]))
                except:
                    print(tree_str.strip())

    if rm_same:
        s2b_list = remove_same(s2b_list)

    return s2b_list


def ptb_to_ae(tree_file):
    with open(tree_file, 'r') as tf:
        ae_list = []
        for tree_str in tf.readlines():
            if len(tree_str.strip()) > 0:
                if not tree_str.strip().startswith("(TOP"):
                    tree_str = "(TOP " + tree_str.strip() + ")"
                try:
                    src, tgt = tree_to_s2b(tree_str.strip())
                    ae_list.append("\t".join([src, src]))
                except:
                    print(tree_str.strip())

    return ae_list


def make_s2b_dataset(train_file, dev_file=None, test_file=None, tgt_dir=None, rm_same=False):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    train = ptb_to_s2b(train_file, rm_same)
    write_docs(fname=os.path.join(tgt_dir, "train.s2b"), docs=train)
    if dev_file is not None:
        dev = ptb_to_s2b(dev_file)
        write_docs(fname=os.path.join(tgt_dir, "dev.s2b"), docs=dev)
    if test_file is not None:
        test = ptb_to_s2b(test_file)
        write_docs(fname=os.path.join(tgt_dir, "test.s2b"), docs=test)


def make_ae_dataset(train_file, dev_file=None, test_file=None, tgt_dir=None, rm_same=False):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    train = ptb_to_ae(train_file)
    write_docs(fname=os.path.join(tgt_dir, "train.ae"), docs=train)
    if dev_file is not None:
        dev = ptb_to_ae(dev_file)
        write_docs(fname=os.path.join(tgt_dir, "dev.ae"), docs=dev)
    if test_file is not None:
        test = ptb_to_ae(test_file)
        write_docs(fname=os.path.join(tgt_dir, "test.ae"), docs=test)


def main_convert(tree_file, out_file=None, mode="s2t"):
    if out_file is None:
        out_file = tree_file + "." + mode
        print("linear tree out is: ", out_file)

    tree_convert_method = tree_convert(mode=mode)

    with open(tree_file, 'r') as tf:
        tree_list = []
        for tree_str in tf.readlines():
            if len(tree_str.strip()) > 0:
                if not tree_str.strip().startswith("(TOP"):
                    tree_str = "(TOP " + tree_str.strip() + ")"
                try:
                    word, tree = tree_convert_method(tree_str.strip())
                    tree_list.append("\t".join([word, tree]))
                except:
                    print(tree_str.strip())
    write_docs(fname=out_file, docs=tree_list)


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--tree_file', dest="tree_file", type=str, help='tree file with Penn TreeBank Format[must]')
    opt_parser.add_argument('--out_file', dest="out_file", type=str, help='output path[optional]')
    opt_parser.add_argument('--mode', dest="mode", type=str, default="s2b", help="linearized tree format:[s2t,s2b,s2s],"
                                                                                 "default is s2b")
    opt = opt_parser.parse_args()
    main_convert(tree_file=opt.tree_file, out_file=opt.out_file, mode=opt.mode)
