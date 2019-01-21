# coding=utf-8

import pickle as pickle

import numpy as np
from nltk.tree import Tree

from .distance_tree import tree2list, get_distance


def to_example(words, token_split=" "):
    return Example(
        src=words.split(token_split),
        tgt=None,
    )


class CachedProperty(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    @staticmethod
    def from_raw_file(file_path, e_type="plain"):
        if e_type == "plain":
            load_class = Example
        elif e_type == 'ptb':
            load_class = PTBExample

        with open(file_path, "r") as f:
            examples = [load_class.parse_raw(line) for line in f]

        return Dataset(examples)

    @staticmethod
    def from_list(raw_list):
        examples = [Example.parse_raw(line) for line in raw_list]
        return Dataset(examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            batch_examples.sort(key=lambda e: -len(e))

            yield batch_examples

    def __sizeof__(self):
        return len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Example(object):
    def __init__(self, src, tgt, idx=0, meta=None, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            self.__setattr__(key, item)

            self.entries.append(key)
        self.src = src
        self.tgt = tgt

        self.idx = idx
        self.meta = meta

    @staticmethod
    def parse_raw(raw_line, field_split='\t', token_split=' '):
        line_items = raw_line.strip().split(field_split)
        if len(line_items) <= 1:
            return Example(
                src=line_items[0].split(token_split),
                tgt=None,
            )
        else:
            return Example(
                src=line_items[0].split(token_split),
                tgt=line_items[1].split(token_split)
            )

    @staticmethod
    def parse_for_control(raw_line, field_split="\t", token_split=" "):
        line_items = raw_line.strip().split(field_split)
        return Example(src=line_items[0].split(token_split), tgt=None), Example(
            src=line_items[1].split(token_split), tgt=None
        )

    def __len__(self):
        return len(self.src)


class PTBExample(Example):
    def __init__(self, src, stag, tags, arcs, distance, tgt=None, idx=0, meta=None):
        super().__init__(src, tgt, idx, meta)
        self.stag = stag
        self.tags = tags
        self.arcs = arcs
        self.distance = distance

    @staticmethod
    def parse_raw(raw_line, field_split='\t', token_split=' '):
        tree = Tree.fromstring(raw_line)
        if tree.label() in ("TOP", "ROOT"):
            assert len(tree) == 1
            tree = tree[0]
        words, stags = zip(*tree.pos())
        linear_trees, arcs, tags = tree2list(tree)

        if type(linear_trees) is str:
            linear_trees = [linear_trees]
        distances_sent, _ = get_distance(linear_trees)
        distances_sent = [0] + distances_sent + [0]

        return PTBExample(
            src=list(words),
            stag=list(stags),
            tags=['<unk>'] + tags + ['<unk>'],
            arcs=['<unk>'] + arcs + ['<unk>'],
            distance=distances_sent
        )

    def __len__(self):
        return super().__len__()
