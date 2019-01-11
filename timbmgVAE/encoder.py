import os

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class SentenceEncoder(nn.Module):
    """
    params:
    vocab: [vocab.src[
    args: contains
        embed_size
        word_dropout
        embedding_dropout
        rnn_type
        hidden_size
    """

    def __init__(self, args, vocab, word_embed=None):
        super(SentenceEncoder, self).__init__()
        self.args = args
        self.vocab = vocab
        self.bidirectional = args.bidirectional
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.hidden_factor = (2 if self.bidirectional else 1) * self.num_layers
        if word_embed is None:
            self.embedding = nn.Embedding(len(vocab), args.embed_size)
        else:
            self.embedding = word_embed

        self.word_dropout_rate = args.word_dropout
        self.embedding_dropout = nn.Dropout(p=args.embedding_dropout)

        if args.rnn_type == 'rnn':
            rnn = nn.RNN
        elif args.rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError()

        self.encoder_rnn = rnn(args.embed_size, args.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)

    def encode(self, input_sents, length):
        batch_size = input_sents.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sents = input_sents[sorted_idx]
        input_embedding = self.embedding(input_sents)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        hidden = hidden.permute(1, 0, 2).contiguous()
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden states
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        return hidden

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


class EncoderVAE(SentenceEncoder):
    def __init__(self, args, vocab, word_embed=None):
        super(EncoderVAE, self).__init__(args, vocab, word_embed)
        self.latent_size = args.latent_size
        self.hidden2mean = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)

    def forward(self, input_sents, length):
        batch_size = input_sents.size(0)
        hidden = self.encode(input_sents, length)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        return mean, logv, z
