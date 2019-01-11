from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pickle
import sys

import numpy as np

from syntaxVAE.vocab import Vocab
from syntaxVAE.vocab import VocabEntry
from utils.dataset import Dataset


def detail(data_set):
    src_vocab = VocabEntry.from_corpus([e.src for e in data_set], )
    tgt_vocab = VocabEntry.from_corpus([e.tgt for e in data_set], )

    vocab = Vocab(src=src_vocab, tgt=tgt_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

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


def prepare_dataset(data_dir, data_dict, tgt_dir, max_src_vocab=16000, max_tgt_vocab=300, vocab_freq_cutoff=1, max_src_length=-1, max_tgt_length=-1,
                    train_size=-1,
                    write_down=True):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    if vocab_freq_cutoff == -1:
        vocab_freq_cutoff = 0
    train_set = length_filter(Dataset.from_raw_file(os.path.join(data_dir, data_dict['train'])), max_src_length, max_tgt_length,
                              max_numbers=train_size)
    dev_set = length_filter(Dataset.from_raw_file(os.path.join(data_dir, data_dict['dev'])), max_src_length, max_tgt_length)
    test_set = length_filter(Dataset.from_raw_file(os.path.join(data_dir, data_dict['test'])), max_src_length, max_tgt_length)

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

    print("Train")
    detail(train_set)
    print("Dev")
    detail(dev_set)
    print("Test")
    detail(test_set)
    if write_down:
        train_file = tgt_dir + "/train.bin"
        dev_file = tgt_dir + "/dev.bin"
        test_file = tgt_dir + "/test.bin"
        vocab_file = tgt_dir + "/vocab.bin"

        pickle.dump(train_set.examples, open(train_file, 'wb'))
        pickle.dump(dev_set.examples, open(dev_file, 'wb'))
        pickle.dump(test_set.examples, open(test_file, 'wb'))
        pickle.dump(vocab, open(vocab_file, 'wb'))
        if 'debug' in data_dict:
            debug_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['debug']))
            debug_file = tgt_dir + "/debug.bin"
            pickle.dump(debug_set.examples, open(debug_file, 'wb'))
