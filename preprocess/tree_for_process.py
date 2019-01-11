# coding=utf-8
"""
convert the ptb to s2b, then write to the data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from utils.tools import write_docs
from utils.tree_linearization import tree_to_s2b


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
