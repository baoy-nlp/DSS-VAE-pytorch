# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pickle
import random
import sys

import numpy as np

from syntaxVAE.vocab import Vocab
from syntaxVAE.vocab import VocabEntry
from utils.config_utils import dict_to_args
from utils.config_utils import yaml_load_dict
from utils.data_helpers import tree2list
from utils.dataset import Dataset
from utils.ptb_utils import load_trees
from utils.tools import write_docs
from utils.tree_linearization import tree_to_s2b


def detail(data_set):
    taget_len = [len(e.tgt) for e in data_set]
    print('Max target len: %d' % max(taget_len), file=sys.stderr)
    print('Avg target len: %d' % np.average(taget_len), file=sys.stderr)

    source_len = [len(e.src) for e in data_set]
    print('Max source len: {}'.format(max(source_len)), file=sys.stderr)
    print('Avg source len: {}'.format(np.average(source_len)), file=sys.stderr)


def data_details(train_list, dev_list, test_list):
    train_set = Dataset.from_list(train_list)
    dev_set = Dataset.from_list(dev_list)
    test_set = Dataset.from_list(test_list)
    src_vocab = VocabEntry.from_corpus([e.src for e in train_set], )
    tgt_vocab = VocabEntry.from_corpus([e.tgt for e in train_set], )

    vocab = Vocab(src=src_vocab, tgt=tgt_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    print("sum info: train:{},dev:{},test:{}".format(
        len(train_set),
        len(dev_set),
        len(test_set),
    ))
    print("Train")
    detail(train_set)
    print("Dev")
    detail(dev_set)
    print("Test")
    detail(test_set)


def prepare_s2b_dataset(data_dir, data_dict, max_src_vocab=16000, max_tgt_vocab=300, vocab_freq_cutoff=1):
    train_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['train']))
    dev_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['dev']))
    test_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['test']))

    # generate vocabulary
    src_vocab = VocabEntry.from_corpus([e.src for e in train_set], size=max_src_vocab, freq_cutoff=vocab_freq_cutoff)
    tgt_vocab = VocabEntry.from_corpus([e.tgt for e in train_set], size=max_tgt_vocab, freq_cutoff=vocab_freq_cutoff)

    vocab = Vocab(src=src_vocab, tgt=tgt_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    print("sum info: train:{},dev:{},test:{}".format(
        len(train_set),
        len(dev_set),
        len(test_set),
    ))
    detail(train_set)
    detail(dev_set)
    detail(test_set)

    train_file = data_dir + "/train.bin"
    dev_file = data_dir + "/dev.bin"
    test_file = data_dir + "/test.bin"
    vocab_file = data_dir + "/origin_vocab.bin"

    pickle.dump(train_set.examples, open(train_file, 'wb'))
    pickle.dump(dev_set.examples, open(dev_file, 'wb'))
    pickle.dump(test_set.examples, open(test_file, 'wb'))
    pickle.dump(vocab, open(vocab_file, 'wb'))
    if 'debug' in data_dict:
        debug_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['debug']))
        debug_file = data_dir + "/debug.bin"
        pickle.dump(debug_set.examples, open(debug_file, 'wb'))


def prepare_raw_data(data_dir, data_dict):
    for key, val in data_dict.items():
        path = os.path.join(data_dir, val)
        data = Dataset.from_raw_file(path)
        out_file = path + ".bin"
        pickle.dump(data.examples, open(out_file, 'wb'))


def prepare_ptb_vocab(vocab, train_file):
    parse_trees = load_trees(path=train_file)
    word_vocab = vocab.src
    tgt_vocab = vocab.tgt
    list_pos = []
    for tree in parse_trees:
        l_words = [item[0] for item in tree.pos()]
        l_tags = [item[1] for item in tree.pos()]
        list_pos.append((l_words, l_tags))
    list_trees = [tree2list(tree) for tree in parse_trees]

    stags = [['<s>'] + list(item[1]) + ['</s>'] for item in list_pos]
    tags = [['<unk>'] + item[2] + ['<unk>'] for item in list_trees]
    arcs = [['<unk>'] + item[1] + ['<unk>'] for item in list_trees]

    stag_vocab = VocabEntry.from_corpus(stags)
    arc_vocab = VocabEntry.from_corpus(corpus=arcs + tags)

    new_vocab = Vocab(
        src=word_vocab,
        tgt=tgt_vocab,
        word=word_vocab,
        stag=stag_vocab,
        arc=arc_vocab
    )
    return new_vocab


def prepare_ptb_to_distance(data_dir, data_dict):
    train_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['train']), e_type='ptb')
    dev_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['dev']), e_type='ptb')
    test_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['test']), e_type='ptb')
    debug_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['debug']), e_type='ptb')

    train_file = data_dir + "/train.bin"
    dev_file = data_dir + "/dev.bin"
    test_file = data_dir + "/test.bin"
    debug_file = data_dir + "/debug.bin"

    pickle.dump(train_set.examples, open(train_file, 'wb'))
    pickle.dump(dev_set.examples, open(dev_file, 'wb'))
    pickle.dump(test_set.examples, open(test_file, 'wb'))
    pickle.dump(debug_set.examples, open(debug_file, 'wb'))


def ptb_to_s2b(tree_file, rm_same=False):
    with open(tree_file, 'r') as tf:
        data_set = []
        for tree_str in tf.readlines():
            src, tgt = tree_to_s2b(tree_str.strip())
            data_set.append("\t".join([src, tgt]))

    if rm_same:
        data_set = remove_same(data_set)

    return data_set


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


def load_ptb_to_s2b(data_dir, data_dict, same_filter=False):
    test_set = ptb_to_s2b(os.path.join(data_dir, data_dict['test']), rm_same=same_filter)
    dev_set = ptb_to_s2b(os.path.join(data_dir, data_dict['dev']), rm_same=same_filter)
    train_set = ptb_to_s2b(os.path.join(data_dir, data_dict['train']), rm_same=same_filter)
    return test_set, dev_set, train_set


def prepare_ptb_to_s2b(data_dir, data_dict, target_dir=None, target_dict=None, max_src=16000, max_tgt=300, cutoff=1, same_filter=False,
                       sample_sub=False):
    test_examples, dev_examples, train_examples = load_ptb_to_s2b(data_dir, data_dicts, same_filter=same_filter)

    if target_dict is None:
        target_dict = {
            'train': "train.s2b",
            "dev": "dev.s2b",
            "test": "test.s2b"
        }
    if target_dir is None:
        target_dir = data_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if sample_sub:
        train_examples, dev_examples, test_examples = sample_sub_snli(test_examples, dev_examples, train_examples)

    write_docs(docs=test_examples, fname=os.path.join(target_dir, target_dict['test']))
    write_docs(docs=dev_examples, fname=os.path.join(target_dir, target_dict['dev']))
    write_docs(docs=train_examples, fname=os.path.join(target_dir, target_dict['train']))
    prepare_s2b_dataset(target_dir, target_dict, max_src, max_tgt, cutoff)


def sample_sub_snli(test_examples, dev_examples, train_examples):
    sub_train = random.sample(train_examples, 90000)
    sub_dev = random.sample(dev_examples, 10000)
    sub_test = random.sample(test_examples, 10000)
    return sub_train, sub_dev, sub_test


"""
process s2b format:
 data_dirs = "../data/s2b"
    data_dicts = {
        'train': "train.s2b",
        'dev': "dev.s2b",
        'test': "test.s2b",
        'debug': "debug.s2b",
    }
prepare_dataset(data_dirs, data_dicts)
"""

"""
convert SNLI format to s2b format and process
config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/snli.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dicts = {
        'train': args.train_file,
        'dev': args.dev_file,
        'test': args.test_file,
    }
    t_dict = {
        'train': args.process_train,
        'dev': args.process_dev,
        'test': args.process_test,
    }

    prepare_ptb_to_s2b(
        data_dir=args.data_dirs,
        data_dict=data_dicts,
        target_dir=args.process_dirs,
        target_dict=t_dict,
        max_src=args.max_src_vocab,
        max_tgt=args.max_tgt_vocab,
        cutoff=args.cut_off,
        same_filter=args.rm_repeat
    )
"""

"""
SNLI DATA DETAILS:

config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/snli.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dicts = {
        'train': args.train_file,
        'dev': args.dev_file,
        'test': args.test_file,
    }

    test, dev, train = load_ptb_to_s2b(
        data_dir=args.data_dirs,
        data_dict=data_dicts,
        same_filter=True,
    )
    print("Filter")
    data_details(train, dev, test)
    test, dev, train = load_ptb_to_s2b(
        data_dir=args.data_dirs,
        data_dict=data_dicts,
        same_filter=False,
    )
    print("Origin")
    data_details(train, dev, test)

"""

if __name__ == '__main__':
    config_file = "/home/user_data/baoy/projects/seq2seq_parser/configs/snli.yaml"
    args_dict = yaml_load_dict(config_file)
    args = dict_to_args(args_dict)
    data_dicts = {
        'train': args.train_file,
        'dev': args.dev_file,
        'test': args.test_file,
    }
    t_dict = {
        'train': args.process_train,
        'dev': args.process_dev,
        'test': args.process_test,
    }

    prepare_ptb_to_s2b(
        data_dir=args.data_dirs,
        data_dict=data_dicts,
        target_dir=args.sample_dirs,
        target_dict=t_dict,
        max_src=args.max_src_vocab,
        max_tgt=args.max_tgt_vocab,
        cutoff=args.cut_off,
        same_filter=args.rm_repeat,
        sample_sub=True,
    )
