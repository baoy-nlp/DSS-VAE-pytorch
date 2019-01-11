import os

import torch
import torch.nn as nn

from submodules.base_seq2seq import BaseSeq2seq
from utils.nn_utils import id2word
from utils.nn_utils import noise_input
from utils.nn_utils import to_input_variable


class AutoEncoder(nn.Module):
    def __init__(self, args, vocab, embed=None):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.vocab = vocab.src
        if embed is None:
            self.embedding = nn.Embedding(len(vocab.src), args.embed_size)
        else:
            self.embedding = embed

        self.word_drop = args.word_drop
        self.eos_id = self.vocab.eos_id
        self.sos_id = self.vocab.sos_id
        self.unk_id = self.vocab.unk_id
        self.pad_id = self.vocab.pad_id

        seq2seq = BaseSeq2seq(
            args=args,
            src_vocab=vocab.src,
            tgt_vocab=vocab.src,
            src_embed=self.embedding,
            tgt_embed=self.embedding if args.share_embed else None,
        )
        self.encoder = seq2seq.encoder
        self.bridger = seq2seq.bridger
        self.decoder = seq2seq.decoder
        self.beam_decoder = seq2seq.beam_decoder

    def encode_var(self, src_var, src_length):
        if self.training and self.word_drop > 0.:
            src_var = noise_input(src_var, droprate=self.word_drop, vocab=self.vocab)
        encoder_outputs, encoder_hidden = self.encoder.forward(input_var=src_var, input_lengths=src_length)
        encoder_hidden = self.bridger.forward(encoder_hidden)
        return encoder_outputs, encoder_hidden

    def encode(self, examples):
        args = self.args
        if isinstance(examples, list):
            src_words = [e.src for e in examples]
        else:
            src_words = examples.src

        src_var = to_input_variable(src_words, self.vocab, cuda=args.cuda, batch_first=True)
        src_length = [len(c) for c in src_words]

        encoder_outputs, encoder_hidden = self.encode_var(src_var, src_length)

        return encoder_hidden

    def init(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def score(self, examples, return_enc_state=False):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        src_words = [e.src for e in examples]
        src_length = [len(c) for c in src_words]

        src_var = to_input_variable(src_words, self.vocab, cuda=args.cuda, batch_first=True)
        tgt_var = to_input_variable(src_words, self.vocab, cuda=args.cuda, append_boundary_sym=True, batch_first=True)
        encoder_outputs, encoder_hidden = self.encode_var(src_var=src_var, src_length=src_length)
        scores = self.decoder.score(inputs=tgt_var, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs)

        enc_states = self.decoder.init_state(encoder_hidden)
        if return_enc_state:
            return scores, enc_states
        else:
            return scores

    def greedy_search(self, examples, to_word=True):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        src_words = [e.src for e in examples]

        src_var = to_input_variable(src_words, self.vocab, cuda=args.cuda, batch_first=True)
        src_length = [len(c) for c in src_words]
        encoder_outputs, encoder_hidden = self.encode_var(src_var=src_var, src_length=src_length)

        decoder_output, decoder_hidden, ret_dict, _ = self.decoder.forward(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            teacher_forcing_ratio=0.0
        )

        result = torch.stack(ret_dict['sequence']).squeeze()
        final_result = []
        example_nums = result.size(1)
        if to_word:
            for i in range(example_nums):
                hyp = result[:, i].data.tolist()
                res = id2word(hyp, self.vocab)
                seems = [[res], [len(res)]]
                final_result.append(seems)
        return final_result

    def batch_greedy_decode(self, examples, to_word=True):
        hidden = self.encode(examples)
        decoder_output, decoder_hidden, ret_dict, _ = self.decoder.forward(
            encoder_hidden=hidden,
            encoder_outputs=None,
            teacher_forcing_ratio=0.0
        )

        result = torch.stack(ret_dict['sequence']).squeeze()
        final_result = []
        example_nums = result.size(1)
        if to_word:
            for i in range(example_nums):
                hyp = result[:, i].data.tolist()
                res = id2word(hyp, self.vocab)
                seems = [[res], [len(res)]]
                final_result.append(seems)
        return final_result

    def beam_search(self, src_sent, beam_size=5, dmts=None):
        if dmts is None:
            dmts = self.args.tgt_max_time_step
        src_var = to_input_variable(src_sent, self.vocab,
                                    cuda=self.args.cuda, training=False, append_boundary_sym=False, batch_first=True)
        src_length = [len(src_sent)]

        encoder_outputs, encoder_hidden = self.encode_var(src_var=src_var, src_length=src_length)

        meta_data = self.beam_decoder.beam_search(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            beam_size=beam_size,
            decode_max_time_step=dmts
        )
        topk_sequence = meta_data['sequence']
        topk_score = meta_data['score'].squeeze()

        completed_hypotheses = torch.cat(topk_sequence, dim=-1)

        number_return = completed_hypotheses.size(0)
        final_result = []
        final_scores = []
        for i in range(number_return):
            hyp = completed_hypotheses[i, :].data.tolist()
            res = id2word(hyp, self.vocab)
            final_result.append(res)
            final_scores.append(topk_score[i].item())
        return final_result, final_scores

    def forward(self, **kwargs):
        raise NotImplementedError

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
