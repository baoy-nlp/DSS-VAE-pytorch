# coding=utf-8
"""
convert the ptb to s2b, then write to the data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocess.tree_convert import make_s2b_dataset
from struct_self.phrase_tree import PhraseTree
from utils.utility import write_docs


def allen_bracker_process(line: str):
    line = line.replace("-LRB- (", "-LRB- -LRB-")
    line = line.replace("-RRB- )", "-RRB- -RRB-")
    line = line.replace("-LRB- [", "-LRB- -LRB-")
    line = line.replace("-RRB- ]", "-RRB- -RRB-")
    return line


def nmt_bracket_process(line: str):
    # line = line.replace("( JJ", "-LRB- -LRB-")
    line = line.replace("（", "(")
    line = line.replace("）", ")")
    line = line.replace("( -LRB-", "-LRB- -LRB-")
    line = line.replace(") -RRB-", "-RRB- -RRB-")
    line = line.replace("[ -LRB-", "-LRB- -LRB-")
    line = line.replace("] -RRB-", "-RRB- -RRB-")
    return line


def switch_symbols(tree):
    if tree.leaf is not None:
        tree.symbol = tree.sentence[tree.leaf][1]
    else:
        for child in tree.children:
            switch_symbols(child)


def preprocess_for_nmt(tree_file):
    tree_list = []
    with open(tree_file, 'r') as tf:
        for tree_str in tf.readlines():
            try:
                tree_str = nmt_bracket_process(tree_str.strip())
                tree = PhraseTree.parse(tree_str)
                sentence = [(item[1], item[0]) for item in tree.sentence]
                tree.propagate_sentence(sentence)
                switch_symbols(tree)
                tree_list.append(tree)
            except:
                print(tree_str)
    write_docs(docs=tree_list, fname=tree_file + ".temp")


def preprocess_for_mt(tree_file):
    tree_list = []
    with open(tree_file, 'r') as tf:
        for tree_str in tf.readlines():
            tree_str = allen_bracker_process(tree_str.strip())
            tree = PhraseTree.parse(tree_str)
            tree_list.append(tree)
    write_docs(docs=tree_list, fname=tree_file + ".temp")


def make_nmt_dataset(train_file, dev_file, test_file, tgt_dir, rm_same=False):
    preprocess_for_nmt(train_file)
    train_file = train_file + ".temp"
    preprocess_for_mt(dev_file)
    dev_file = dev_file + ".temp"
    preprocess_for_mt(test_file)
    test_file = test_file + ".temp"

    make_s2b_dataset(train_file, dev_file, test_file, tgt_dir, rm_same)


def make_nmt_simple_dataset(train_file, dev_file, test_file, tgt_dir, rm_same=False):
    preprocess_for_mt(train_file)
    train_file = train_file + ".temp"
    preprocess_for_mt(dev_file)
    dev_file = dev_file + ".temp"
    preprocess_for_mt(test_file)
    test_file = test_file + ".temp"

    make_s2b_dataset(train_file, dev_file, test_file, tgt_dir, rm_same)
