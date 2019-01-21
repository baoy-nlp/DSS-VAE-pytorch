"""
Non-Autoregressive Auto-Encoder Decoding Model.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import get_decoder
# from decoder.parallel_decoder import MatrixDecoder
from encoder import get_encoder
from utils.nn_funcs import to_input_variable
from utils.nn_funcs import to_target_word
from utils.nn_funcs import unk_replace


class ParallelAE(nn.Module):
    def __init__(self, args, vocab, embed=None):
        super(ParallelAE, self).__init__()

        self.args = args
        self.vocab = vocab

        self.encoder = get_encoder(args=args,
                                   vocab_size=len(self.vocab.src),
                                   model=args.enc_type,
                                   embed=embed,
                                   pad=self.vocab.src.pad_id
                                   )
        self.max_len = args.tgt_max_time_step
        # self.decoder = MatrixDecoder(
        #     vocab_size=len(self.vocab.tgt),
        #     max_len=args.src_max_time_step,
        #     input_dim=self.encoder.out_dim,
        #     hidden_dim=args.enc_inner_hidden,
        #     mapper_dropout=args.dropm,
        #     out_dropout=args.dropo,
        # )

        self.decoder = get_decoder(
            args=args,
            input_dim=self.encoder.out_dim,
            vocab_size=len(self.vocab.tgt),
            model=args.dec_type,
            pad=self.vocab.tgt.pad_id
        )

        # self.word_drop = args.word_drop
        self.word_drop = 0.0
        # self.normalization = 1.0
        # self.norm_by_words = False
        # self.critic = SequenceCriterion(padding_idx=-1)

    def encode(self, seqs_x, seqs_length=None):
        if self.training and self.word_drop > 0.:
            seqs_x = unk_replace(seqs_x, dropoutr=self.word_drop, vocab=self.vocab.src)
        if self.args.enc_type == "att":
            enc_ret = self.encoder.forward(seqs_x)
            enc_hid = enc_ret['out']
        else:
            enc_hid, _ = self.encoder.forward(seqs_x, input_lengths=seqs_length)
        return enc_hid.mean(dim=1)

    def decode(self, encoder_output, seqs_y=None):
        return self.decoder.forward(inputs=encoder_output, seqs_y=seqs_y)

    def forward(self, seqs_x, seqs_y=None, x_length=None, to_word=True, log_prob=True):
        """
        Args:
            seqs_x: previous seqs_y
            seqs_y:
            x_length:
            to_word:
            log_prob:
        """
        enc_out = self.encode(seqs_x, seqs_length=x_length)
        # check: batch_size, hidden_size
        prob = self.decode(enc_out, seqs_y=seqs_y)
        # check: batch_size, max_len, vocab_size

        if log_prob:
            prob = F.log_softmax(prob, dim=-1)
        if to_word:
            return {
                "prob": prob,
                "pred": to_target_word(prob, vocab=self.vocab.tgt)
            }
        else:
            return {
                "prob": prob,
            }

    def score(self, examples, return_enc_state=False):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        src_words = [e.src for e in examples]
        tgt_words = [e.tgt for e in examples]

        src_length = [len(c) for c in src_words]
        seqs_x = to_input_variable(src_words, self.vocab.src, cuda=args.cuda, batch_first=True)
        seqs_y = to_input_variable(tgt_words, self.vocab.tgt, max_len=self.max_len,
                                   cuda=args.cuda,
                                   append_boundary_sym=True,
                                   batch_first=True)
        # log_probs = self.forward(seqs_x, seqs_y, x_length=src_length, to_word=False)['prob']
        # # y_label = seqs_y[:, 1:].contiguous()
        # y_label = seqs_y.contiguous()
        #
        # words_norm = y_label.ne(-1).float().sum(1)
        #
        # loss = self.critic(inputs=log_probs, labels=y_label, reduce=False, normalization=self.normalization)
        #
        # if self.norm_by_words:
        #     loss = loss.div(words_norm).sum()
        # else:
        #     loss = loss.sum()
        # return loss
        enc_out = self.encode(seqs_x, seqs_length=src_length)
        # check: batch_size, hidden_size
        return self.decoder.score(raw_score=enc_out, tgt_var=seqs_y)

    def predict(self, examples, to_word=True):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        src_words = [e.src for e in examples]
        src_length = [len(c) for c in src_words]
        seqs_x = to_input_variable(src_words, self.vocab.src, cuda=args.cuda, batch_first=True)
        predict = self.forward(seqs_x, x_length=src_length, to_word=to_word)
        return predict['pred']

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict)

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
        }

        torch.save(params, path)

    @staticmethod
    def load(load_path):
        params = torch.load(load_path, map_location=lambda storage, loc: storage)
        args = params['args']
        vocab = params['vocab']
        model = ParallelAE(args, vocab)
        model.load_state_dict(params['state_dict'])
        if args.cuda:
            model = model.cuda()
        return model
