from __future__ import print_function

import sys

sys.path.append(".")
from collections import Counter
from itertools import chain

import torch

from utils.math_utils import js_divergence, kl_divergence
from utils.ops import get_tensor
from metrics.base_metric import BaseEvaluator


# sys.path.append(".")


class CorpusDistribution(object):
    @staticmethod
    def get_unigram_distribution(examples, vocab):
        """
        :param examples: list of sentence
        :param vocab:
        :return:
        """
        unigram_count = [0] * len(vocab)
        word_freq = Counter(chain(*examples))

        for word in word_freq:
            unigram_count[vocab[word]] = word_freq['word']
        count = get_tensor(unigram_count)
        count += (1.0 / torch.sum(count)) * (count.eq(0.0).float())
        # count += 1e-6
        return count / torch.sum(count)


class UnigramKLEvaluator(BaseEvaluator):
    def _check_format(self, **kwargs):
        pass

    def __init__(self, ):
        super(UnigramKLEvaluator, self).__init__(name="Unigram KL")

    def get_evaluate(self, corpus_source, pred_source, vocab, dtype='js'):
        """
        :param corpus_source: list of sentence
        :param pred_source: list of sentence
        :param vocab: VocabularyEntry
        :param dtype: "js" or "kl"
        :return:
        """
        ref_dis = CorpusDistribution.get_unigram_distribution(examples=corpus_source, vocab=vocab)
        pre_dis = CorpusDistribution.get_unigram_distribution(examples=pred_source, vocab=vocab)
        func = js_divergence if dtype == 'js' else kl_divergence
        return func(ref_dis, pre_dis)


if __name__ == "__main__":
    train_path = "/home/user_data/baoy/projects/seq2seq_parser/data/snli-sample/train.bin"
    dev_path = "/home/user_data/baoy/projects/seq2seq_parser/data/snli-sample/dev.bin"
    test_path = "/home/user_data/baoy/projects/seq2seq_parser/data/snli-sample/test.bin"

    vocab_file = "/home/user_data/baoy/projects/seq2seq_parser/data/snli-sample/origin_vocab.bin"

    plain_file = "./gen.text"
    with open(plain_file, 'r') as f:
        sample = [line.split(" ") for line in f.readlines()]
    from utils.dataset import Dataset
    from syntaxVAE.vocab import Vocab

    vocab = Vocab.from_bin_file(vocab_file)
    train_exam = Dataset.from_bin_file(train_path).examples
    train = [e.src for e in train_exam]

    dev_exam = Dataset.from_bin_file(dev_path).examples
    dev = [e.src for e in dev_exam]

    test_exam = Dataset.from_bin_file(test_path).examples
    test = [e.src for e in test_exam]

    t = UnigramKLEvaluator()
    # print("train with dev:", t.get_evaluate(train, dev, vocab.src))
    # print("train with test:", t.get_evaluate(train, test, vocab.src))
    # print("dev with test", t.get_evaluate(dev, test, vocab.src))
    # print("test with dev", t.get_evaluate(test, dev, vocab.src))
    print("train with sample", t.get_evaluate(test, sample, vocab.src))
