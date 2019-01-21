from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys

sys.path.append(".")
import numpy as np

from struct_self.dataset import Dataset
from struct_self.vocab import Vocab
from struct_self.vocab import VocabEntry


def data_details(train_set, dev_set, test_set):
    def detail(data_set):
        _src_vocab = VocabEntry.from_corpus([e.src for e in data_set], )
        _tgt_vocab = VocabEntry.from_corpus([e.tgt for e in data_set], )

        _vocab = Vocab(src=_src_vocab, tgt=_tgt_vocab)
        print('generated vocabulary %s' % repr(_vocab), file=sys.stderr)

        _target_len = [len(e.tgt) for e in data_set]
        print('Max target len: %d' % max(_target_len), file=sys.stderr)
        print('Avg target len: %d' % np.average(_target_len), file=sys.stderr)

        source_len = [len(e.src) for e in data_set]
        print('Max source len: {}'.format(max(source_len)), file=sys.stderr)
        print('Avg source len: {}'.format(np.average(source_len)), file=sys.stderr)

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


def length_filter(dataset, max_src_len=-1, max_tgt_len=-1, max_numbers=-1):
    examples = dataset.examples
    if max_src_len != -1:
        new_examples = []
        for x in examples:
            if len(x.src) < max_src_len:
                new_examples.append(x)
        examples = new_examples
    if max_tgt_len != -1:
        new_examples = []
        for x in examples:
            if len(x.src) < max_tgt_len:
                new_examples.append(x)
        examples = new_examples
    if max_numbers != -1:
        from random import sample
        train_idx = sample(range(len(examples)), max_numbers)
        examples = [examples[idx] for idx in train_idx]
    dataset.examples = examples
    return dataset


def prepare_dataset(data_dir, data_dict, tgt_dir, max_src_vocab=16000, max_tgt_vocab=300, vocab_freq_cutoff=1,
                    max_src_length=-1, max_tgt_length=-1,
                    train_size=-1,
                    write_down=True):
    train_pair = os.path.join(data_dir, data_dict['train'])
    dev_pair = os.path.join(data_dir, data_dict['dev'])
    test_pair = os.path.join(data_dir, data_dict['test'])

    make_dataset(train_pair, dev_pair, test_pair, tgt_dir, max_src_vocab, max_tgt_vocab, vocab_freq_cutoff,
                 max_src_length,
                 max_tgt_length, train_size,
                 write_down)


def make_dataset(train_raw, dev_raw, test_raw=None, tgt_dir=".", max_src_vocab=16000, max_tgt_vocab=300,
                 vocab_freq_cutoff=1,
                 max_src_length=-1, max_tgt_length=-1,
                 train_size=-1,
                 write_down=True):
    if vocab_freq_cutoff == -1:
        vocab_freq_cutoff = 0
    train_set = length_filter(Dataset.from_raw_file(train_raw), max_src_length,
                              max_tgt_length,
                              max_numbers=train_size)
    dev_set = length_filter(Dataset.from_raw_file(dev_raw), max_src_length,
                            max_tgt_length)
    if test_raw is not None:
        test_set = length_filter(Dataset.from_raw_file(test_raw), max_src_length,
                                 max_tgt_length)
    else:
        test_set = dev_set

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

    data_details(train_set, dev_set, test_set)

    if write_down:
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
        train_file = tgt_dir + "/train.bin"
        dev_file = tgt_dir + "/dev.bin"
        test_file = tgt_dir + "/test.bin"
        vocab_file = tgt_dir + "/vocab.bin"

        pickle.dump(train_set.examples, open(train_file, 'wb'))
        pickle.dump(dev_set.examples, open(dev_file, 'wb'))
        pickle.dump(test_set.examples, open(test_file, 'wb'))
        pickle.dump(vocab, open(vocab_file, 'wb'))
        # if 'debug' in data_dict:
        #     debug_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['debug']))
        #     debug_file = tgt_dir + "/debug.bin"
        #     pickle.dump(debug_set.examples, open(debug_file, 'wb'))


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--train_file', dest="train_file", type=str,
                            help='train file after process with tree_convert output file')
    opt_parser.add_argument('--dev_file', dest="dev_file", type=str, help='dev file with src,tgt pair')
    opt_parser.add_argument('--test_file', dest="test_file", type=str, help='test file with src,tgt pair')
    opt_parser.add_argument('--tgt_dir', dest="tgt_dir", type=str, help='target dir')
    opt_parser.add_argument("--max_src_vocab", dest="max_src_vocab", type=int, default=16000,
                            help="source phrase vocab size, default is 16000")
    opt_parser.add_argument("--max_tgt_vocab", dest="max_tgt_vocab", type=int, default=300,
                            help="target phrase vocab size, 300 for parse")
    opt_parser.add_argument("--vocab_freq_cutoff", dest="vocab_freq_cutoff", type=int, default=-1,
                            help="sort freq of word in train set, "
                                 "and cutoff which freq which lower than this value, default is -1")
    opt_parser.add_argument("--max_src_len", dest="max_src_len", type=int, default=-1,
                            help="max length of example 's source input , default is -1")
    opt_parser.add_argument("--max_tgt_len", dest="max_tgt_len", type=int, default=-1,
                            help="max length of example 's target output , default is -1")
    opt_parser.add_argument("--train_size", dest="train_size", type=int, default=-1,
                            help="the number of examples select from whole dataset, default is -1, means all")

    opt = opt_parser.parse_args()

    make_dataset(
        train_raw=opt.train_file,
        dev_raw=opt.dev_file,
        test_raw=opt.test_file,
        tgt_dir=opt.tgt_dir,
        max_src_vocab=opt.max_src_vocab,
        max_tgt_vocab=opt.max_tgt_vocab,
        max_src_length=opt.max_src_len,
        max_tgt_length=opt.max_tgt_len,
        vocab_freq_cutoff=opt.vocab_freq_cutoff,
        train_size=opt.train_size,
    )
